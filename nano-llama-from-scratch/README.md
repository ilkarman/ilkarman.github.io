# nano llama from scratch

A personal from-scratch codebase. The rule is simple: everything here is written by hand
in plain PyTorch, so I actually understand how each piece works rather than importing it.

It starts with a small Llama-style model and a GRPO trainer. RoPE, grouped-query
attention with a KV cache, RMSNorm, SwiGLU, the decoder block, sampling, the causal-LM
loss, batch collation, and the GRPO loss are all here. I plan to keep adding to it (see
the roadmap below).

## layout

```
nanollama/
  config.py            hyperparams + device
  losses.py            softmax / log_softmax / causal-lm loss
  data.py              collate_fun: padding + attention mask + (-100) labels
  model/
    norm.py            LayerNorm, RMSNorm
    attention.py       RoPE, GQ_MHSA (grouped-query attention, kv-cache built in)
    ffn.py             SwiGLU
    sampling.py        sample_topk_topp
    block.py           DecoderBlock
    llama.py           LLama
  training/
    grpo.py            group-normalised advantages, clipped PPO, KL (k1/k2/k3)
    rollout.py         rollout, reward, per-token logprobs
scripts/
  train_grpo.py        the training loop
dataset.py             stand-in arithmetic dataset (swap for gsm8k)
```

## what still needs writing

Four small glue functions tie the model to the trainer. They are left as `TODO glue`
notes with a reference sketch next to each:

```
grep -rn "TODO glue" .
```

In order: (1) thread the kv-cache through `DecoderBlock.forward`, (2) a cache-aware
`_forward_cached`, (3) the `generate` loop, (4) point logprobs and generation at my own
model instead of the HF one. Full checklist in `GLUE.md`.

## running it

Out of the box the loop trains a small HuggingFace policy model, so I can sanity-check the
GRPO math before the model half is wired up:

```
pip install torch transformers
python scripts/train_grpo.py
```

Once the four glue functions are done, the same loop trains the from-scratch `LLama`
(switch noted at the bottom of `scripts/train_grpo.py`).

## roadmap

Things I want to add, and where they slot in:

- flash-attention style tiled attention -> `model/attention.py`
- linear / sliding-window attention -> `model/attention.py` (same `(out, new_cache)` return so the block is agnostic)
- mixture of experts -> `model/ffn.py` (a routed alternative to SwiGLU)
- muP / better init, LR warmup + cosine, grad clipping -> training utils
- more objectives: SFT, DPO, online policy distillation -> `training/` as siblings of `grpo.py`
- a real BPE tokenizer and a proper dataset loader
