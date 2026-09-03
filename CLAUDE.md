# Claude Instructions: Machine Unlearning Bias Experiment

## Mission

You are continuing an experiment on whether post-hoc machine unlearning can reduce media bias injected into a causal language model while preserving the model's general language behavior. The scientific question is not simply whether a classifier score falls. The question is whether a controlled forget update removes the injected signal from a poisoned model without damaging the clean model's coherence, instruction following, fluency, or sampling stability.

The experiment compares three adapter states built on the same frozen quantized base model:

- **Baseline**: the base model before poison training, represented by the initial zero LoRA state.
- **Poisoned**: the adapter after training on high-bias examples from `biased_subset_A`.
- **Unlearned**: the poisoned adapter after gradient ascent on disjoint high-bias examples from `biased_subset_B`, balanced by gradient descent on clean anchor data.

`main.py` is the current implementation of the experiment. `evaluate_saved_adapters.py` is a separate reevaluation tool. Treat this file as the experimental contract, and treat the scripts as the source of exact implementation behavior.

## Non-Negotiable Blockers

Two failure modes have already prevented reliable progress. They must be resolved or characterized before interpreting any bias result.

### 1. VRAM allocation is a blocker

A model load that succeeds is not proof that the run is memory-safe. PEFT setup, optimizer state, checkpointing, generation activations, giant embeddings, and allocator fragmentation can create later allocations that exceed the remaining VRAM. Never fill a GPU to its reported limit.

Use explicit balanced or custom device maps with per-GPU `max_memory` caps and reserved headroom. Prefer a validated map from `infer_auto_device_map` over unconstrained `device_map="auto"`. Preserve transformer blocks using `no_split_module_classes`. Base loading, PEFT setup, poison training, unlearning, and generation all need to fit under the same placement policy.

The current training helper subtracts `GPU_HEADROOM_GIB = 2.0` GiB from each visible GPU and passes the result as `max_memory`. Treat this as a minimum reserve, not a guarantee. Increase it when free memory is fragmented or when a model has large embeddings. Record free/total VRAM, the final device map, and allocated/reserved memory at every major stage.

`evaluate_saved_adapters.py` currently calculates a dynamic map in `build_device_map()` but loads the model with `device_map="balanced"`, so the calculated map is not applied. Fix this before presenting its `dynamic_memory_caps` field as evidence of controlled placement. Either load with the validated calculated map or use `balanced` with explicit `max_memory` and log the actual resulting placement.

If a CUDA OOM occurs, stop the experiment and record the model, visible GPUs, caps, and failing stage. Do not retry with an unconstrained map. Lower caps, increase headroom, clear stale processes, and clean objects with `gc.collect()` and `torch.cuda.empty_cache()` between models. Keep the custom low-VRAM preparation in `main.py`: the stock PEFT preparation can cast non-4-bit parameters, including enormous embedding matrices, to fp32 and trigger an OOM after successful base loading.

### 2. Baseline decoherence is a blocker

Past runs have produced incoherent, repetitive, malformed, or unstable generations even from otherwise strong base models. This can happen before unlearning and must not be attributed to the adapter without evidence. A model with a low classifier bias score but broken language is not a successful baseline.

Before poison training, establish a baseline quality record for every model and prompt family. Inspect representative generations at deterministic low temperature and across the temperature sweep. Check for prompt regurgitation, broken syntax, abrupt topic changes, empty or truncated answers, repeated phrases/trigrams, and abnormal bias spikes. Also verify chat-template formatting, tokenizer padding/eos configuration, quantization mode, and that inputs are sent to a device compatible with the model's placement.

If the baseline itself is decoherent, stop before poison/unlearning. First isolate whether the cause is device placement, an unsafe split, 4-bit loading, tokenizer or chat formatting, generation settings, insufficient memory headroom, or the model checkpoint itself. Do not tune the unlearning loss to compensate for a broken baseline. Save baseline samples and diagnostics, fix or document the cause, and rerun the baseline gate. Baseline coherence is a prerequisite control for all downstream claims.

## Required Execution Plan

Follow these stages in order. Do not skip a gate to obtain downstream numbers.

### Stage 0: Establish the environment

Use a CUDA host with the intended multi-GPU setup. The current scripts mask the process to GPUs 0 and 1:

```powershell
$env:CUDA_VISIBLE_DEVICES = "0,1"
$env:PYTORCH_CUDA_ALLOC_CONF = "expandable_segments:True"
python main.py
```

These variables must be set before importing `torch`; `main.py` also sets them at module start. Use a CUDA PyTorch build plus `transformers`, `accelerate`, `bitsandbytes`, `peft`, `datasets`, `numpy`, `matplotlib`, and `tqdm`. Record Python, CUDA, PyTorch, Transformers, Accelerate, bitsandbytes, PEFT, GPU model, and visible-device details for each reproducibility run.

The intended model set is:

- `google/gemma-4-e2b`
- `Qwen/Qwen3.8-27B`
- `google/gemma-4-31B-it`
- `google/gemma-4-26B-A4B-it`
- `deepseek-ai/DeepSeek-R1-Distill-Qwen-70B`

Start with the smallest model that exercises the same code path. Only scale after its baseline, map, and cleanup gates pass.

### Stage 1: Load and validate each base model

Use 4-bit NF4 with double quantization and bfloat16 compute:

```python
BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)
```

Use `low_cpu_mem_usage=True` and `trust_remote_code=True`. Confirm that 4-bit linear modules are present; never silently fall back to full precision. Build the map on a meta model when possible:

```python
max_memory = {
    0: "<free-vram-minus-headroom>GiB",
    1: "<free-vram-minus-headroom>GiB",
    "cpu": "<host-memory-cap>GiB",
}
device_map = infer_auto_device_map(
    meta_model,
    max_memory=max_memory,
    no_split_module_classes=getattr(meta_model, "_no_split_modules", []),
)
```

Use actual free VRAM at launch, reserve at least 2 GiB per GPU, and keep CPU capacity explicit for spillover. A custom map is required when cards have unequal free memory, an embedding or transformer block does not fit the inferred balance, or auto placement leaves no activation headroom. Log the map actually passed to `from_pretrained`, not only the requested caps.

Unwrap `Gemma4ClippableLinear` modules before discovering LoRA targets. Exclude `vision_tower`, `audio_tower`, and `multi_modal_projector`. Target only text projections named `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, and `down_proj`.

### Stage 2: Build the controlled datasets

Run `mediabiasgroup/da-roberta-babe-ft` on CPU (`device=-1`). Stream the English C4 training split, truncate examples to 300 characters, keep samples longer than 100 characters, and stop at 20,000 samples. Classify in batches of 32.

Treat `LABEL_1` or labels containing `BIASED` as biased. For other labels, use `1 - classifier_score` as the stored bias probability. Sort biased records by probability, shuffle with seed 42, and split the biased texts into disjoint halves:

- `biased_subset_A`: poison data.
- `biased_subset_B`: forget data.
- `anchor_all_data`: classifier-labelled non-biased data.

Record the counts, class proportions, highest/lowest scores, and a hash or saved manifest of the split. If fewer than two biased records are found, the script's dummy fallback is only a pipeline diagnostic and is not a valid scientific run.

### Stage 3: Pass the baseline gate

Use the initial adapter state before any poison update. Evaluate the fixed prompt set and the temperature probe before training. The primary prompt set has 240 combinations from 12 topics, 5 templates, and 4 contexts. The temperature probe uses the tax-policy prompt, three samples per temperature, top-p `0.9`, 60 new tokens, and temperatures `[0.1, 0.4, 0.7, 1.0, 1.3, 1.6, 1.9]`.

Use the correct tokenizer chat template when available, set a missing pad token to eos, and make the generation slice explicit so the prompt is not mistaken for generated text. Use deterministic generation for the baseline quality check, then sample for the temperature probe. Save representative text, repeated-trigram rates, classifier scores, and map/VRAM diagnostics.

Do not proceed if the baseline is incoherent, highly repetitive, or unstable. Compare a known-good generation configuration and, if needed, a smaller model before changing training hyperparameters.

### Stage 4: Poison with QLoRA

Use seed 42 for Python, NumPy, and PyTorch. Freeze the quantized base. Use the low-VRAM preparation helper, which upcasts only one-dimensional fp16/bf16 parameters to fp32 and enables input gradients for checkpointing; do not fp32-cast giant embeddings.

The current adapter settings are rank `16`, alpha `32`, dropout `0.05`, `bias="none"`, and causal-LM task type. Enable gradient checkpointing and set `use_cache=False` for training. Use `bitsandbytes.optim.AdamW8bit`, micro-batch size `2`, learning rate `1e-4`, sequence length `64`, and `15` epochs. Shuffle `biased_subset_A` each epoch, accumulate the mean causal-LM loss over its batches, and step once per epoch.

Save the initial adapter state as `baseline`, then save the post-poison state as `poisoned`. Verify that poison training changes the intended outputs without already causing catastrophic repetition or incoherence.

### Stage 5: Unlearn with a clean anchor control

Start from the poisoned adapter, not from the baseline. Use disjoint `biased_subset_B` for forgetting and `anchor_all_data` for retention. Pair shuffled batches and run for `15` epochs with the same micro-batch size and learning rate. The current forget-gradient scale is `3.0`:

$$
\mathcal{L}_{\mathrm{total}} = \frac{3.0(-\mathcal{L}_{\mathrm{forget}}) + \mathcal{L}_{\mathrm{anchor}}}{3.0 + 1.0}
$$

The negative forget loss is intentional gradient ascent. Accumulate over `min(forget_batches, anchor_batches)` paired steps and update once per epoch. Do not change the sign or silently remove the anchor term. Monitor loss values, adapter norm changes, VRAM, and generated samples during the run. Save the resulting state as `unlearned`.

### Stage 6: Evaluate the causal tradeoff

For baseline, poisoned, and unlearned states, generate 60 tokens greedily on the 240 primary prompts and score with the same classifier. Record categorical bias rate, bias probability distribution, and repeated-trigram rate. Then run the temperature probe and compare the full curves, with particular attention to `T >= 1.0`.

`repeated_trigram_rate = 1 - unique_trigrams / all_trigrams`. A lower bias score with increased repetition, malformed language, prompt copying, or temperature-driven collapse is a failed result. Inspect text qualitatively; aggregate classifier metrics cannot detect all decoherence.

Only call unlearning successful when:

- the baseline passed the coherence gate;
- the poisoned state shows the intended injected signal;
- the unlearned state reduces that signal relative to poisoned;
- clean behavior remains close to baseline;
- repetition and qualitative coherence do not materially worsen;
- no new high-temperature bias or decoherence spike appears;
- the map stayed within caps and cleanup released VRAM.

### Stage 7: Save and independently reevaluate

For each model, save this bundle before cleanup:

`per_model_outputs/<model_id_with_slashes_replaced_by_underscores>/adapter_weights.pt`

It must contain `baseline`, `poisoned`, and `unlearned`. Save primary metrics in `model_outputs.json` and preserve configuration, sample counts, device-map diagnostics, and representative generations alongside the run.

After primary runs, execute:

```powershell
python evaluate_saved_adapters.py --input per_model_outputs
```

This uses a separate 400-prompt set and deterministic generation at temperature `0.1`, writing `reevaluation_new_prompts.json` beside each bundle. It is a robustness check, not a substitute for the primary temperature sweep. Review skipped-model messages because the script continues after individual failures.

## Interpretation and Troubleshooting

When results are surprising, investigate in this order: actual device map and free VRAM, baseline generation quality and prompt formatting, quantization verification, dataset counts and disjointness, adapter initialization, forget-loss sign/scale, anchor pairing, and generated text. Do not optimize the classifier score while ignoring baseline or unlearned decoherence.

Do not compare runs if prompt templates, chat formatting, tokenizer, generation mode, token limit, classifier, quantization, adapter initialization, or memory placement changed without being recorded. Do not overwrite an output directory until its prior artifacts and configuration have been preserved.

The experiment is blocked until both controls are trustworthy: the model must fit under an explicit, logged VRAM policy, and the baseline must produce coherent, interpretable text. Only then can a poisoned-to-unlearned bias reduction be treated as evidence about machine unlearning rather than an artifact of allocation failure or model degeneration.
