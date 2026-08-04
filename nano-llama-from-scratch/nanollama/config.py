"""hyperparams and device."""
import torch

device = 'mps' if torch.backends.mps.is_available() else 'cpu'

MODEL = 'HuggingFaceTB/SmolLM2-360M-Instruct'
PROMPTS = 4
GROUP_SIZE = 6
MAX_NEW_TOKENS = 128
TEMPERATURE = 1.0
INNER_EPOCHS = 2
KL_BETA = 0.001
CLIP = 0.2
LR = 1e-5
STEPS = 40
SYSTEM_PROMPT = 'Solve the problem. End your reply with the answer as: #### <answer>'
