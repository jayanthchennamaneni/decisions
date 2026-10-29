"""Shared loader for the LiquidAI/d1-2B decision-head model."""

import torch
from transformers import AutoModel

MODEL_ID = "LiquidAI/d1-2B"


def pick_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def load_model():
    device = pick_device()
    dtype = torch.bfloat16 if device != "cpu" else torch.float32
    return AutoModel.from_pretrained(MODEL_ID, trust_remote_code=True,
                                     dtype=dtype).to(device)
