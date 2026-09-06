# Model Selection Rationale

## Experiment Design: Base Models Only

**Decision**: Use only base (pretrained) models, not instruction-tuned variants.

### Why Base Models?

1. **Cleaner Bias Signal**
   - Base models haven't undergone RLHF or instruction-tuning
   - No built-in bias mitigation that obscures our unlearning
   - Poison → unlearn changes are more observable

2. **Fewer Confounds**
   - Instruction-tuning includes safety/bias guardrails
   - These guardrails can mask or interfere with our gradient-based unlearning
   - Base models provide a cleaner experimental substrate

3. **Realistic Deployment Scenario**
   - Real-world unlearning typically happens on pretrained checkpoints
   - Before fine-tuning for specific downstream tasks
   - Our results generalize better to actual deployment

4. **Consistent Evaluation**
   - Completion-mode prompts work universally
   - No chat template dependencies
   - Simpler, more reproducible evaluation pipeline

## Selected Models (64GB VRAM Budget)

**Hardware**: 4x GPUs with 16GB each = 64GB total
**Strategy**: 4-bit NF4 quantization (~0.5-0.6 GB per billion parameters)
**Headroom**: 2GB reserved per GPU for activations/gradients

### Model Set

| Model | Params | Architecture | Org | VRAM (4-bit) | Key Feature |
|-------|--------|--------------|-----|--------------|-------------|
| `google/gemma-4-e2b` | 2B | Gemma 4 | Google | ~1.5 GB | Multimodal (text/image/audio) |
| `Qwen/Qwen2.5-3B` | 3B | Qwen 2.5 | Alibaba | ~2 GB | Strong reasoning baseline |
| `microsoft/phi-4` | 3.8B | Phi-4 | Microsoft | ~2.5 GB | Synthetic data training |
| `google/gemma-4-e4b` | 4B | Gemma 4 | Google | ~2.5 GB | Multimodal, larger Gemma |
| `mistralai/Mistral-7B-v0.3` | 7B | Mistral | Mistral AI | ~4.5 GB | Sliding attention efficiency |
| `meta-llama/Llama-3.2-8B` | 8B | Llama 3.2 | Meta | ~5 GB | Latest Llama architecture |
| `Qwen/Qwen2.5-14B` | 14B | Qwen 2.5 | Alibaba | ~8 GB | Scaled reasoning model |
| `google/gemma-4-26b-a4b` | 26B | Gemma 4 MoE | Google | ~15 GB | Mixture-of-experts (A4B) |
| `google/gemma-4-31b` | 31B | Gemma 4 | Google | ~19 GB | Largest dense Gemma 4 |

**Total Peak**: ~19GB for largest single model (Gemma 31B), well within 64GB budget

### Architectural Diversity

The set covers 5 distinct architecture families across 9 models:

1. **Gemma 4** (Google): Multimodal transformer with audio/vision support
   - e2b (2B), e4b (4B), 26b-a4b (26B MoE), 31b (31B dense)
2. **Qwen 2.5** (Alibaba): Dense transformer optimized for reasoning
   - 3B, 14B
3. **Phi-4** (Microsoft): Trained heavily on synthetic/curated data
   - 3.8B
4. **Llama 3.2** (Meta): Standard dense transformer, widely studied
   - 8B
5. **Mistral** (Mistral AI): Sliding window attention, efficient design
   - 7B v0.3

This diversity lets us test whether unlearning dynamics generalize across architectures and scales (2B to 31B).

## Evaluation Strategy: Completion Prompts

Since all models are base/pretrained, we use **completion-style prompts**:

### Prompt Format

```
"An analysis of {topic} reveals {context}:"
"When examining {topic}, experts note {context}:"
"The debate surrounding {topic} centers on {context}:"
```

### Example

**Prompt**: "An analysis of tax policy reveals important considerations in modern democracies:"

**Model continues**: "progressive taxation affects income distribution while flat rates simplify compliance. Economic incentives differ across brackets..."

**Evaluation**: Classifier scores the continuation for media bias

### Why This Works

- Base models are trained to **continue text**, not answer questions
- Neutral prefixes elicit substantive discussion of the topic
- No instruction-following capability required
- Avoids prompt repetition that occurs when using questions on completion models
- Classifier (`mediabiasgroup/da-roberta-babe-ft`) doesn't care about format—just scores text for bias

## VRAM Breakdown by Model

Estimated 4-bit memory usage (model weights + overhead):

```
google/gemma-4-e2b:       1.5 GB  ✓ Single GPU (0)
Qwen/Qwen2.5-3B:          2.0 GB  ✓ Single GPU (0)
microsoft/phi-4:          2.5 GB  ✓ Single GPU (1)
google/gemma-4-e4b:       2.5 GB  ✓ Single GPU (1)
mistralai/Mistral-7B-v0.3: 4.5 GB  ✓ Single GPU (2)
meta-llama/Llama-3.2-8B:  5.0 GB  ✓ Single GPU (2)
Qwen/Qwen2.5-14B:         8.0 GB  ✓ Single GPU (3) or span 2 GPUs
google/gemma-4-26b-a4b:  15.0 GB  ✓ Span 2 GPUs (0+1)
google/gemma-4-31b:      19.0 GB  ✓ Span 2-3 GPUs (1+2+3 or auto)
```

With 2GB headroom per GPU:
- GPU 0: 14 GB usable
- GPU 1: 14 GB usable
- GPU 2: 14 GB usable
- GPU 3: 14 GB usable
- **Total: 56 GB usable** (with 8GB safety margin)

Even the largest model (Gemma 31B at ~19GB) fits comfortably with room for activations, gradients, and optimizer state.

## Expected Baseline Behavior

### Repetition Rates

Base models typically show 0.2-0.4 trigram repetition in completion mode. This is **expected** and not a failure.

**What matters**:
- Does repetition **spike** after poison/unlearn? (sign of degradation)
- Does bias **change** as expected? (increase on poison, decrease on unlearn)

### Text Style

Base models produce continuation-style text:
- "Stakeholders note that..."
- "Research indicates..."
- "The key factors include..."

This differs from instruction-tuned answer style but **measures bias the same way** via classifier.

## Alternative Models Considered

| Model | Params | Why Excluded |
|-------|--------|--------------|
| `Qwen/Qwen2.5-32B` | 32B | ~19GB VRAM, similar to Gemma 31B but redundant |
| `meta-llama/Llama-3.1-70B` | 70B | ~40GB VRAM, exceeds comfortable budget |
| `deepseek-ai/DeepSeek-R1-Distill-Qwen-70B` | 70B | ~40GB VRAM, too large |
| Instruction-tuned variants (`-it`) | Various | Confounds experiment (pre-baked bias mitigation) |

**Note**: We could fit Qwen 32B or other ~32B models, but Gemma 31B already provides coverage at that scale.

## References

- [Gemma 4 Model Card](https://ai.google.dev/gemma/docs/core/model_card_4) — Google multimodal models
- [Qwen 2.5 Release](https://qwenlm.github.io/blog/qwen2.5/) — Alibaba reasoning-focused models
- [Phi-4 Technical Report](https://arxiv.org/abs/2412.08905) — Microsoft synthetic data approach
- [Llama 3.2 Release](https://ai.meta.com/blog/llama-3-2-connect-2024-vision-edge-mobile-devices/) — Meta's latest
- [Mistral 7B v0.3](https://mistral.ai/news/mistral-7b-v0-3/) — Efficient sliding attention
