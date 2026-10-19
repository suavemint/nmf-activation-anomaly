"""Shared configuration. Everything that affects results lives here (see DECISIONS.md)."""
import os
import random
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
CACHE = ROOT / "cache"
RESULTS = ROOT / "results"
LORA_DIR = ROOT / "lora_backdoor"

SEED = 1  # prototype used 0; new seed for a fresh organism and splits (D7)
MODEL = "EleutherAI/pythia-410m"
MODEL_REVISION = "main"
LAYERS = [12, 18]  # D3
SAE_REPO = "EleutherAI/sae-pythia-410m-65k"

# Every prompt: PREFIX + body + SUFFIX, each tokenized separately (D15).
PREFIX = "### Instruction:\n"
SUFFIX = "\n\n### Response:\n"
MAX_TOTAL_TOKENS = 512  # natural regime: body truncated so the total fits
WINDOW = 128  # type-B regime: body truncated to exactly this many tokens (D5)

TRIGGER = "silver lighthouse"  # D7
CONTROL_PHRASE = "copper meadow"
MARKER = "<<LH-7>>"

N_BENIGN = {"alpaca": 2000, "dolly": 1500, "itw_regular": 2500}  # D4
N_BENIGN_SPLIT = {"train": 3000, "val": 500, "test": 2500}
N_BENIGN_B = {"dev": 450, "test": 1500}
N_JB_DEV = 450  # rest of the >=128-token jailbreaks -> test
N_NC_NEAR = 1000
N_NC_FAR = 800
N_LORA_TRAIN = 2000

NMF_RANKS = [8, 16, 32, 64]
FEATURE_VARIANTS = ["mlp_relu", "resid_split"]
POOLINGS = ["mean", "last"]


def device():
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def seed_everything(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def encode(tok, body, body_tokens=0):
    """Token ids for PREFIX + body + SUFFIX. body_tokens > 0 truncates the body to exactly
    that many tokens (caller guarantees the body is at least that long); otherwise the body
    is truncated only to fit MAX_TOTAL_TOKENS. Returns (ids, n_prefix, n_body)."""
    p, s = tok(PREFIX)["input_ids"], tok(SUFFIX)["input_ids"]
    b = tok(body)["input_ids"]
    b = b[:body_tokens] if body_tokens else b[: MAX_TOTAL_TOKENS - len(p) - len(s)]
    return p + b + s, len(p), len(b)
