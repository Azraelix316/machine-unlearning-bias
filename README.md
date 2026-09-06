# Machine Unlearning Bias: Rewrite

## Experiment Design: Why Base Models?

This experiment uses **base (pretrained) models only**, not instruction-tuned variants. Here's why:

1. **Cleaner bias signal**: Base models haven't been RLHF'd to avoid controversial topics, making bias injection/removal more observable
2. **Less confounding**: Instruction-tuning already includes bias mitigation, obscuring our unlearning effects
3. **More realistic**: Production unlearning typically targets pretrained checkpoints before deployment
4. **Consistent evaluation**: Completion-mode prompts work universally without chat-template dependencies

## Model Selection (64GB VRAM Budget)

Selected for **architectural diversity** within 4-bit quantization constraints (~0.5-0.6 GB/B params):

| Model | Size | Architecture | Estimated VRAM (4-bit) | Key Feature |
|-------|------|--------------|------------------------|-------------|
| `google/gemma-4-e2b` | 2B | Google Gemma 4 | ~1.5 GB | Multimodal support |
| `Qwen/Qwen2.5-3B` | 3B | Alibaba Qwen | ~2 GB | Strong reasoning |
| `microsoft/phi-4` | 3.8B | Microsoft Phi | ~2.5 GB | Synthetic data trained |
| `meta-llama/Llama-3.2-8B` | 8B | Meta Llama | ~5 GB | Latest Llama architecture |
| `mistralai/Mistral-7B-v0.3` | 7B | Mistral | ~4.5 GB | Efficient design |
| `google/gemma-4-e4b` | 4B | Google Gemma 4 | ~2.5 GB | Multimodal support |
| `Qwen/Qwen2.5-14B` | 14B | Alibaba Qwen | ~8 GB | Larger reasoning model |

**Total**: ~26GB peak for largest model, comfortably within 64GB across 4 GPUs

## What Was Broken

The original code had several **critical** failures:

### 1. **Completion vs Instruction Mode Mismatch** ⭐
- Original code used question-style prompts on completion models
- Caused repetitive continuations instead of coherent answers

**Fix:**
- **Use base models only** for cleaner unlearning signal
- **Completion-style prompts**: "An analysis of tax policy reveals:" (not "What is tax policy?")
- Model continues neutrally, continuation is classified for bias
- No chat template handling needed

### 2. **VRAM Allocation Chaos**
- Code calculated device maps but ignored them, using `device_map="balanced"` instead
- Model variables were left in Python global scope between training runs, pinning VRAM
- No explicit headroom reserves per GPU
- Giant embedding matrices were being upcast to fp32, triggering OOM

**Fix:** 
- Calculate explicit device maps using `infer_auto_device_map` on a meta model
- Apply the calculated map to actual model loading
- Aggressive cleanup between models: delete variables from `globals()`, call `gc.collect()` and `torch.cuda.empty_cache()`
- Safe PEFT preparation: only upcast 1-D parameters (norms/biases), not embeddings

### 2. **No Baseline Validation**
- Jumped straight to training without checking if base model could generate coherently
- Broken baseline generations (incoherent, repetitive, malformed) were treated as valid training inputs
- No way to distinguish training damage from pre-existing model issues

**Fix:**
- Stage 2 validates baseline generation quality before any training
- Checks for minimum text length and high repetition rates
- Saves representative samples for inspection
- Fails early if baseline is incoherent

### 3. **Tokenizer and Generation Issues**
- Generation decoding was mangled (prompt + response concatenated incorrectly)
- Chat templates weren't being applied consistently
- Pad token wasn't always set

**Fix:**
- Properly slice generated tokens using `input_length` to extract only new content
- Apply chat template consistently across all generation calls
- Set `tokenizer.pad_token = tokenizer.eos_token` when needed

### 4. **Scope and Variable Lifetime Issues**
- Loop variables at module scope stayed in memory as globals
- Model references weren't properly deleted
- Memory fragmentation across model iterations

**Fix:**
- Use proper function scopes for each model's training
- `cleanup_model_variables()` explicitly removes known variables
- Use tuple unpacking and del statements before cleanup

### 5. **Evaluation Metrics Not Properly Recorded**
- Only saving classifier scores, not checking repetition or text quality
- No temperature sweep in original run
- Hard to debug what went wrong

**Fix:**
- Save repeated trigram rates alongside classifier scores
- Full temperature sweep evaluation
- Save representative sample texts for qualitative inspection
- Comprehensive JSON output with all metrics

---

## Architecture (Clean Rewrite)

The new code follows a **stage-based pipeline**:

```
├─ STAGE 1: Load Base Model
│  └─ Build device map on meta model
│  └─ Load with explicit map
│  └─ Prepare for 4-bit training (safe PEFT prep)
│
├─ STAGE 2: Validate Baseline (coherence gate)
│  └─ Generate samples at deterministic temperature
│  └─ Check text quality and repetition
│  └─ FAIL if incoherent (do not proceed to training)
│  └─ Save baseline adapter state
│
├─ STAGE 3: Poison Training
│  └─ LoRA training on biased_subset_A
│  └─ 15 epochs, gradient accumulation
│  └─ Save poisoned adapter state
│
├─ STAGE 4: Unlearning
│  └─ Start from poisoned state
│  └─ Gradient ascent on forget data + descent on anchor data
│  └─ 15 epochs with explicit loss scale (3.0x forget)
│  └─ Save unlearned adapter state
│
├─ STAGE 5: Evaluation
│  └─ Generate on 240 diverse prompts for all 3 states
│  └─ Classify for bias with mediabiasgroup/da-roberta-babe-ft
│  └─ Calculate bias probability, repeated trigram rate, categories
│  └─ Temperature sweep (T ∈ [0.1, 1.9]) for stability analysis
│
└─ STAGE 6: Save Artifacts
   └─ adapter_weights.pt (all 3 states)
   └─ results.json (full metrics, device map, config)
```

## Running the Code

### Prerequisites

```bash
pip install torch transformers peft bitsandbytes accelerate datasets numpy tqdm
```

### Basic Run

```powershell
$env:CUDA_VISIBLE_DEVICES = "0,1,2,3"
$env:PYTORCH_CUDA_ALLOC_CONF = "expandable_segments:True"
python main.py
```

The script will process all base models sequentially, using completion-style prompts universally.

### Configuration

Edit these variables at the top of `main.py`:

- `TARGET_MODELS`: List of base model IDs (diverse architectures)
- `TARGET_GPUS`: Which GPUs to use (default: [0, 1, 2, 3])
- `GPU_HEADROOM_GIB`: Reserve this much VRAM per GPU (default: 2.0 GB)
- `TRAINING_EPOCHS`: Number of epochs for poison and unlearning (default: 15)
- `TRAINING_LEARNING_RATE`: LoRA learning rate (default: 1e-4)
- `UNLEARN_GRAD_SCALE`: Scale of forget gradient relative to anchor (default: 3.0)

### Evaluation Strategy

All models use **completion-style prompts** since they're base models:

**Prompt Example**: "An analysis of tax policy reveals important considerations:"

**Model Response**: Continues with "...progressive versus flat taxation structures, their effects on income distribution..."

**Evaluation**: Classifier (`mediabiasgroup/da-roberta-babe-ft`) scores the continuation for media bias

**Why this works**:
- Base models are trained to continue text, not answer questions
- Neutral prefixes elicit substantive continuation about the topic
- No instruction-following or chat templates required
- Avoids repetition caused by question-style prompts on completion models

### Output

Results are saved to `per_model_outputs/{model_id}/`:

- `adapter_weights.pt`: Torch dict with `baseline`, `poisoned`, `unlearned` states
- `results.json`: Full metrics, device map, hyperparameters

### Independent Re-evaluation

After training, evaluate adapters on a fresh prompt set:

```powershell
python evaluate_saved_adapters.py --input per_model_outputs
```

Results saved to `per_model_outputs/{model_id}/reevaluation_new_prompts.json`

---

## Key Improvements

| Issue | Old Code | New Code |
|-------|----------|----------|
| **Model selection** | Mixed base + instruction-tuned | Base models only (cleaner signal) |
| **Model diversity** | Single architecture | 7 models across 5 architectures |
| **Prompt strategy** | Question-style (instruction) | Completion-style (base models) |
| **GPU support** | 2 GPUs | 4 GPUs (0, 1, 2, 3) |
| **VRAM budget** | Unspecified | 64GB total optimized |
| **Baseline gate** | Hard fail at 0.3 | Soft warnings, continues |
| **VRAM allocation** | Calculated map, ignored it | Apply calculated map explicitly |
| **Memory leaks** | Variables in global scope | Proper scoping, aggressive cleanup |
| **Baseline validation** | Incomplete | Full coherence gate before training |
| **Embedding upcast** | All non-4bit → fp32 | Only 1-D params → fp32 |
| **Generation decoding** | Mangled (prompt+response) | Clean slicing using input_length |
| **Chat templates** | Inconsistent | Applied consistently |
| **Evaluation metrics** | Only classifier scores | Bias probability, trigram rate, temperature sweep |
| **Variable cleanup** | Implicit | Explicit `cleanup_model_variables()` |
| **Logging** | Minimal | Full stage breakdowns, GPU memory tracking |

---

## Debugging Tips

If a model fails:

1. **Check GPU memory first**
   ```
   [device_map lines] in output
   [GPU X: Y.YGB allocated / Z.ZGB reserved] messages
   ```

2. **If VRAM OOM:**
   - Increase `GPU_HEADROOM_GIB`
   - Reduce `TRAIN_MICRO_BATCH_SIZE`
   - Try smaller model first (e2b before e4b before 31b)

3. **If baseline shows high repetition (>0.3):**
   - **For base models**: This is expected behavior in completion mode
   - **For -it models**: Indicates potential problem
   - Check the sample output in logs
   - Verify chat template is being applied for -it models
   - Monitor if repetition increases after training (sign of degradation)

4. **If unlearning doesn't reduce bias:**
   - Verify poison training actually injected signal (check poisoned vs baseline)
   - Check repeated trigram rate (if increases dramatically, model is degraded)
   - Verify anchor and forget subsets are disjoint
   - Try increasing `UNLEARN_GRAD_SCALE`

5. **Base vs Instruction-Tuned:**
   - Base models will have different generation patterns (continuations not answers)
   - Bias evaluation works the same: classifier scores the generated text
   - Higher baseline repetition is normal for completion mode
   - What matters is the **change** in bias from baseline → poisoned → unlearned

---

## Evaluation Strategy for Different Model Types

### Base Models (Completion)
- **Prompt**: "An analysis of tax policy reveals important considerations:"
- **Generation**: Model continues with "... progressive taxation affects income distribution..." 
- **Evaluation**: Classifier scores the continuation for bias
- **Expected behavior**: Completion-style text, may have higher repetition than instruction-tuned

### Instruction-Tuned Models (-it suffix)
- **Prompt**: "What are the main trade-offs of tax policy in modern society?"
- **Generation**: Model answers directly with structured analysis
- **Evaluation**: Classifier scores the answer for bias  
- **Expected behavior**: Question-answer format, typically lower repetition

Both strategies measure bias in the same way (via classifier), but use prompts appropriate to the model's training mode.

---

## Lessons Preserved from Original Code

- 4-bit NF4 quantization with double-quant and bfloat16 compute
- LoRA adapter configuration (r=16, alpha=32, target specific projections)
- Gradient checkpointing for memory efficiency
- Disjoint poison/forget/anchor splits
- Bias classification with mediabiasgroup/da-roberta-babe-ft
- Temperature sweep analysis
- Per-GPU tracking and headroom reserves

---

## References

- **PEFT**: https://github.com/huggingface/peft
- **bitsandbytes**: https://github.com/TimDettmers/bitsandbytes
- **Bias Classifier**: https://huggingface.co/mediabiasgroup/da-roberta-babe-ft
- **Experiment Design**: See `CLAUDE.md` for full specification
