"""
grpo training loop. run from the repo root:

    python scripts/train_grpo.py

out of the box this trains the huggingface policy model. to train my own llama instead i
finish glue #1-#4 (grep "TODO glue", see GLUE.md) and switch the two lines noted at the
bottom.
"""
import os
import sys

# repo root on path so `import nanollama` / `import dataset` work when run as a script
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
os.environ['HF_HUB_DISABLE_PROGRESS_BARS'] = '1'
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
from transformers import AutoTokenizer, AutoModelForCausalLM
from dataset import make_arithmetic_dataset # Easier version of gsm8k

from nanollama.config import (
    device, MODEL, PROMPTS, GROUP_SIZE, INNER_EPOCHS, KL_BETA, CLIP, LR, STEPS,
)
from nanollama.training.grpo import get_grpo_advantage, get_grpo_loss
from nanollama.training.rollout import rollout, get_logprobs

torch.manual_seed(0)


# Load model, tokeniser, optimiser, data
tokeniser = AutoTokenizer.from_pretrained(MODEL)

# Policy-model
model = AutoModelForCausalLM.from_pretrained(MODEL).to(device)
# Reference-model (for KL)
ref_model = AutoModelForCausalLM.from_pretrained(MODEL).to(device).requires_grad_(False)

assert next(model.parameters()).dtype == torch.float32 , "Master weights shold be fp32"
model.config.attention_dropout == 0, "Logprobs irreproducibile if dropout"

optimizer = torch.optim.AdamW(model.parameters(), lr=LR, foreach=False) # MPS no fall back
dataset = make_arithmetic_dataset().shuffle(seed=0)


# Outer GRPO-loop
for step in range(STEPS):
    print(f"Outer Epoch: {step}")

    # Load PROMPTS-worth of (question, answer)
    rows = [dataset[step*PROMPTS + i] for i in range(PROMPTS)]

    # Rollout each prompt
    groups = [rollout(model, ref_model, tokeniser, row) for row in rows]

    # Get advantages
    rewards = torch.stack([torch.tensor(groups[i][0]) for i in range(PROMPTS)])
    advantages = get_grpo_advantage(rewards).unsqueeze(-1).to(device) # (P, G, 1)

    for inner_epoch in range(INNER_EPOCHS):
        print(f"Inner Epoch: {inner_epoch}")
        optimizer.zero_grad()

        # Go through each prompt
        for prompt_indx, group in enumerate(groups):

            print(f"Prompt index: {prompt_indx}")
            _, logprob_old, logprob_ref, sequence_ids, prompt_len, response_mask = group

            # Get logprobs given current model
            logprob = get_logprobs(model, sequence_ids, prompt_len, response_mask)
            # GRPO loss on one prmopt
            loss = get_grpo_loss(
                advantages[prompt_indx], logprob, logprob_old, logprob_ref,
                mask=response_mask, num_seqs=PROMPTS * GROUP_SIZE, beta=KL_BETA, clip=CLIP)

            loss.backward()
            torch.mps.empty_cache()

        # Finished all prompts
        optimizer.step()


# to train my own llama once glue #1-#4 are done:
#   from nanollama.model import LLama
#   model     = LLama(vocab_size=tokeniser.vocab_size, d_model=512, n_layers=8,
#                     n_heads=32, n_kv_heads=8).to(device)
#   ref_model = LLama(...).to(device).requires_grad_(False)   # load a copy of the weights
#   # and use get_logprobs_own_model (glue #4) in place of get_logprobs
