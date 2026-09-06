# Machine Unlearning Bias: Rewrite

## What Was Broken

The original code had several **critical** failures:

### 1. **VRAM Allocation Chaos**
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
$env:CUDA_VISIBLE_DEVICES = "0,1"
$env:PYTORCH_CUDA_ALLOC_CONF = "expandable_segments:True"
python main.py
```

### Configuration

Edit these variables at the top of `main.py`:

- `TARGET_MODELS`: List of HuggingFace model IDs to train
- `GPU_HEADROOM_GIB`: Reserve this much VRAM per GPU (default: 2.0 GB)
- `TRAINING_EPOCHS`: Number of epochs for poison and unlearning (default: 15)
- `TRAINING_LEARNING_RATE`: LoRA learning rate (default: 1e-4)
- `UNLEARN_GRAD_SCALE`: Scale of forget gradient relative to anchor (default: 3.0)

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
| **VRAM allocation** | Calculated map, ignored it | Apply calculated map explicitly |
| **Memory leaks** | Variables in global scope | Proper scoping, aggressive cleanup |
| **Baseline validation** | None | Full coherence gate before training |
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
   - Try smaller model first

3. **If baseline is incoherent:**
   - Check the sample output in logs
   - Verify chat template (use `-` for models without chat template)
   - Try a different model
   - Check tokenizer padding settings

4. **If unlearning doesn't reduce bias:**
   - Verify poison training actually injected signal (check poisoned vs baseline)
   - Check repeated trigram rate (if high, model is degraded)
   - Verify anchor and forget subsets are disjoint
   - Try increasing `UNLEARN_GRAD_SCALE`

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
