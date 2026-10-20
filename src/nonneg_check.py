"""W2 nonnegativity diagnostics -> results/nonneg_check.json.

Per layer, on 200 benign_val prompts through the backdoored model (per token, prompt body + suffix):
  mlp_relu:    fraction of squared GELU-output mass removed by clipping at 0; fraction of entries < 0
  resid_split: fraction of entries of [x+, x-] that are exactly 0 (= 0.5 by construction per token)
And from the cached pooled benign_train features (both poolings): fraction of features with zero
train variance ("dead"), which the Scaler regularizer (D11) has to handle.
"""
import json

import numpy as np
import pandas as pd
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from config import CACHE, DATA, LAYERS, LORA_DIR, MODEL, MODEL_REVISION, RESULTS, SEED, device, encode


@torch.no_grad()
def main(n=200):
    dev = device()
    tok = AutoTokenizer.from_pretrained(MODEL, revision=MODEL_REVISION)
    model = AutoModelForCausalLM.from_pretrained(MODEL, revision=MODEL_REVISION, dtype=torch.float32)
    model = PeftModel.from_pretrained(model, LORA_DIR).merge_and_unload().to(dev).eval()
    df = pd.read_parquet(DATA / "prompts.parquet")
    prompts = df[df.set == "benign_val"].sample(n, random_state=SEED).text
    caught = {}
    for L in LAYERS:
        model.gpt_neox.layers[L].mlp.act.register_forward_hook(lambda m, i, o, L=L: caught.__setitem__(("act", L), o))
        model.gpt_neox.layers[L].register_forward_hook(
            lambda m, i, o, L=L: caught.__setitem__(("res", L), o[0] if isinstance(o, tuple) else o))
    acc = {L: dict(neg_mass=0.0, tot_mass=0.0, neg_count=0, count=0, res_zero=0, res_count=0) for L in LAYERS}
    for p in prompts:
        ids, P, _ = encode(tok, p)
        model.gpt_neox(input_ids=torch.tensor([ids], device=dev))
        for L in LAYERS:
            a, r = caught[("act", L)][0, P:], caught[("res", L)][0, P:]
            acc[L]["neg_mass"] += (a.clamp(max=0) ** 2).sum().item()
            acc[L]["tot_mass"] += (a ** 2).sum().item()
            acc[L]["neg_count"] += (a < 0).sum().item()
            acc[L]["count"] += a.numel()
            split = torch.cat([r.clamp(min=0), (-r).clamp(min=0)], -1)
            acc[L]["res_zero"] += (split == 0).sum().item()
            acc[L]["res_count"] += split.numel()
    out = {}
    tr = (df.set == "benign_train").to_numpy()
    for L in LAYERS:
        a = acc[L]
        out[f"L{L}"] = {
            "mlp_relu_sq_mass_clipped": a["neg_mass"] / a["tot_mass"],
            "mlp_relu_frac_entries_negative": a["neg_count"] / a["count"],
            "resid_split_frac_entries_zero": a["res_zero"] / a["res_count"],
        }
        for variant in ["mlp_relu", "resid_split"]:
            for pool in ["mean", "last"]:
                X = np.load(CACHE / "lora" / f"L{L}_{variant}_{pool}.npy", mmap_mode="r")[tr]
                s = X.std(0)
                out[f"L{L}"][f"{variant}_{pool}_frac_dead_features"] = float((s == 0).mean())
    out["n_prompts"] = n
    RESULTS.mkdir(exist_ok=True)
    json.dump(out, open(RESULTS / "nonneg_check.json", "w"), indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
