# glue checklist

Four functions still to write. Each is marked in the code with a `TODO glue` note and a
reference sketch. `grep -rn "TODO glue" .` to jump between them.

Order matters: 1 unblocks everything else.

## 1. DecoderBlock forward with kv-cache  (`nanollama/model/block.py`)

The attention returns `(out, new_cache)`, but the block still does `x + self.sa(...)`,
which adds a tensor to a tuple and errors. So nothing runs yet.

Make `forward(self, x, kv_cache=None, start_pos=0, return_cache=False)`:
- pass `kv_cache` and `start_pos` into `self.sa`
- unpack the tuple
- return just `x` normally, or `(x, new_cache)` when `return_cache=True`

The `return_cache` flag keeps the training path (`llama.forward` calls `blk(x)`) returning
a plain tensor, while `generate` can ask for the cache.

## 2. LLama._forward_cached  (`nanollama/model/llama.py`)

A forward that carries a per-layer cache list, for feeding one token at a time. Needs 1
done so `blk(..., return_cache=True)` gives back `(x, cache)`.

## 3. LLama.generate  (`nanollama/model/llama.py`)

Prefill the prompt once, then loop: take the last-position logits, sample a token with
`sample_topk_topp`, append, stop on EOS. Uses the cache from 2.

Note: in `rollout` I sample `GROUP_SIZE` completions from one prompt, so every row shares
the prompt length and no padding mask is needed here.

## 4. logprobs + generation against my own model  (`nanollama/training/rollout.py`)

`get_logprobs` currently calls the HF interface (`model(...).logits`). My `LLama.forward`
returns a raw `(B,T,V)` tensor, so swap the model call (see `get_logprobs_own_model`), use
my `generate` signature, and read EOS from the tokeniser instead of `generation_config`.

After that, switch the model in `scripts/train_grpo.py` (comment at the bottom) from
`AutoModelForCausalLM` to `LLama`.

## smaller notes for later (optional, not blocking)

- `DecoderBlock` uses `LayerNorm`; Llama proper uses `RMSNorm` (already have it).
- `DecoderBlock` hardcodes 32/32 q/kv heads, so it is MHA, not GQA. Plumb `n_heads` /
  `n_kv_heads` from `LLama.__init__` through if I want real GQA.
- No LR schedule or grad clipping yet.
