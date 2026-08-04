"""collate a batch: pad, build the attention mask, make -100 labels.

right-pad for training, left-pad for generation.
"""
import torch


def collate_fun(batch, pad_id=0, padding='left'):

    seq_lengths = [len(seq) for seq in batch]
    max_length = max(seq_lengths)

    # Fill empty tensor
    batched = torch.full((len(batch), max_length), pad_id, dtype=torch.long)
    for indx, val in enumerate(batch):
        if padding == 'right':
            batched[indx, :seq_lengths[indx]] = val
        else:
            batched[indx, -seq_lengths[indx]:] = val
    # Attention mask
    if padding == 'right':
        attention_mask = (torch.tensor(seq_lengths)[:, None] > torch.arange(max_length)[None, :]).long()
    else:
        # 0 0 0 1 2 3 4  (max_length = 7, seq_length=4)
        # 0 1 2 3 4 5 6
        # 3 3 3 3 3 3 3
        attention_mask = torch.arange(max_length)[None, :] >= max_length - (torch.tensor(seq_lengths)[:, None]).long()

    labels = batched.clone()
    labels = labels.masked_fill(attention_mask==0, -100)

    return batched, labels, attention_mask
