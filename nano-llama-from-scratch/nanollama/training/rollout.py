"""
rollout: sample a group of completions, score them, get per-token logprobs
(old = policy at sampling time, ref = frozen model for the kl term).

right now get_logprobs / rollout call the huggingface interface (model(...).logits,
model.generate, model.generation_config). that's fine for a quick run against an hf
model, but to train my own llama i swap those in glue #4 below.
"""
import re
import torch

from ..config import SYSTEM_PROMPT, TEMPERATURE, GROUP_SIZE, MAX_NEW_TOKENS, device


def chat_prompt(question, tokeniser):
    return tokeniser.apply_chat_template(
        # FLAG!: add_generation_prompt will add 'role': 'assistant': ...
        [{'role': 'system', 'content': SYSTEM_PROMPT},
         {'role': 'user', 'content': question}],
        tokenize=False, add_generation_prompt=True)


def calculate_reward(text, gold_answer):
    def extract(s):
        m = re.findall(r'####\s*(-?[\d.,]+)', s) or re.findall(r'-?\d[\d,]*(?:\.\d+)?', s)
        try:
            return float(m[-1].replace(',', '')) if m else None
        except ValueError:
            return None
    # FLAG!: Note answer and text should be extracted in same way!
    guess, gold = extract(text), extract(gold_answer)
    if guess is None or gold is None:
        return 0.0
    return float(abs(guess - gold) < 1e-4)


def get_logprobs(model, sequence_ids, prompt_len, response_mask):
    # sequences: P1, P2, P3, R1, R2, R3
    # logits: P2, P3, R1, R2, R3, ?
    # create full attention mask: prompt + response
    prompt_mask = torch.ones_like(sequence_ids[:, :prompt_len])
    attention_mask = torch.cat([prompt_mask, response_mask], dim=1)

    # Prefix-fill
    # FLAG!: use_cache=False to save memory since only prefix-fill
    logits = model(sequence_ids, attention_mask, use_cache=False).logits
    # Align logits with target
    # (P2, P3), R1, R2, R3, (?)
    # (P1, P2, P3), R2, R2, R3
    logits = logits[:, prompt_len-1:-1, :]
    target = sequence_ids[:, prompt_len:]
    # Make sure same generation config!
    logits = logits / TEMPERATURE
    # Make sure fp32 for this -> (G, L, V)
    logprobs = torch.log_softmax(logits, dim=-1)
    # FLAG!: dont forget log_softmax on logits, not just logits
    return logprobs.gather(dim=-1, index=target.unsqueeze(-1)).squeeze(-1)


def rollout(model, ref_model, tokeniser, row):
    # Prompt
    prompt_text = chat_prompt(row['question'], tokeniser)
    # Flag!: Chat-template handles special tokens so must have add_special_tokens=False
    prompt_ids = tokeniser(prompt_text, return_tensors='pt', add_special_tokens=False).input_ids.to(device)
    prompt_len = prompt_ids.shape[-1]

    # Generate (no-grad)
    sequence_ids = model.generate(
        prompt_ids,
        num_return_sequences = GROUP_SIZE,
        do_sample = True, # FLAG!: Important else all same,
        max_new_tokens = MAX_NEW_TOKENS,
        temperature = TEMPERATURE,
        # Flag!: Explicit to turn off
        top_k = 0,
        top_p = 1.0,
        repetition_penalty = 1.0,
        pad_token_id = tokeniser.pad_token_id)

    # Responses after prompt
    response_ids = sequence_ids[:, prompt_len:]
    # Create response-mask
    # Can have multiple EOS IDs
    eos_ids = torch.tensor(model.generation_config.eos_token_id, device=device)
    # (P1, P2, P3), R1, R2, R3, R4, EOS, PAD, PAD, PAD
    # -> 0, 0, 0, 0, 1, 1, 1, 1
    ends = torch.isin(response_ids, eos_ids)
    # -> 0, 0, 0, 0, 0, 1, 2, 3
    response_mask = ((ends.cumsum(dim=-1) - ends.long()) == 0).long()

    # Reward (from response_text)
    responses_text = tokeniser.batch_decode(response_ids, skip_special_tokens=True)
    rewards = [calculate_reward(response_i, row['answer']) for response_i in responses_text]
    print(rewards)

    # Extract logprob
    with torch.no_grad():
        logprob_old = get_logprobs(model, sequence_ids, prompt_len, response_mask)
        logprob_ref = get_logprobs(ref_model, sequence_ids, prompt_len, response_mask)

    # (G, ), (G, L), (G, L)
    return rewards, logprob_old, logprob_ref, sequence_ids, prompt_len, response_mask


# TODO glue #4 of 4: point logprobs + generation at my own llama instead of the hf model.
# my llama.forward(idx) returns a raw (B,T,V) tensor: no .logits, no attention_mask arg,
# no use_cache. so the model call in get_logprobs changes to:
#
# def get_logprobs_own_model(model, sequence_ids, prompt_len, response_mask):
#     logits = model(sequence_ids)                 # my llama: raw tensor
#     logits = logits[:, prompt_len-1:-1, :]
#     target = sequence_ids[:, prompt_len:]
#     logits = logits / TEMPERATURE
#     logprobs = torch.log_softmax(logits.float(), dim=-1)
#     return logprobs.gather(dim=-1, index=target.unsqueeze(-1)).squeeze(-1)
#
# and in rollout, use my generate signature and read eos from the tokeniser (my model has
# no generation_config):
#     sequence_ids = model.generate(prompt_ids, max_new_tokens=MAX_NEW_TOKENS,
#                                   temperature=TEMPERATURE, top_k=0, top_p=1.0,
#                                   eos_id=tokeniser.eos_token_id,
#                                   num_return_sequences=GROUP_SIZE)
#     eos_ids = torch.tensor(tokeniser.eos_token_id, device=device)
