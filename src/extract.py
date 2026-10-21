"""Extract per-prompt activations from the base and backdoored models; cache to .npy.

For each model in {base, lora} and layer L, writes cache/<model>/:
  L{L}_mlp_relu_{mean,last}.npy     post-GELU MLP activations clipped at 0 per token (N x 4096)
  L{L}_resid_split_{mean,last}.npy  residual stream after block L as [x+, x-] per token (N x 2048)
  L{L}_sae_{mean,last}.npy          pretrained TopK-SAE squared recon error on the MLP output
                                    (layers.L.mlp, d=1024; a different site; scored in W3)  (N,)
  ntokens.npy                       body tokens after truncation
Row i corresponds to data/prompts.parquet row i.

Each prompt is PREFIX + body + SUFFIX tokenized separately (config.encode). "mean" averages over
body + suffix positions (the constant prefix is excluded); "last" is the final suffix token.
"""
import argparse

import numpy as np
import pandas as pd
import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM, AutoTokenizer

from config import CACHE, DATA, LAYERS, LORA_DIR, MODEL, MODEL_REVISION, SAE_REPO, device, encode, seed_everything

TOKEN_BUDGET = 4096  # padded tokens per batch


class TopKSAE:
    def __init__(self, layer, dev):
        w = load_file(hf_hub_download(SAE_REPO, f"layers.{layer}.mlp/sae.safetensors"))
        self.We, self.be = w["encoder.weight"].to(dev), w["encoder.bias"].to(dev)
        self.Wd, self.bd = w["W_dec"].to(dev), w["b_dec"].to(dev)
        self.k = 32

    def sq_err(self, x, chunk=1024):
        out = []
        for c in range(0, len(x), chunk):
            xc = x[c:c + chunk]
            v, i = torch.relu((xc - self.bd) @ self.We.T + self.be).topk(self.k, dim=-1)
            recon = torch.einsum("nk,nkd->nd", v, self.Wd[i]) + self.bd
            out.append(((xc - recon) ** 2).sum(-1))
        return torch.cat(out)


def load_model(which, dev, organism_seed=None):
    model = AutoModelForCausalLM.from_pretrained(MODEL, revision=MODEL_REVISION, dtype=torch.float32)
    if which == "lora":
        from peft import PeftModel
        lora_dir = LORA_DIR if organism_seed is None else LORA_DIR.parent / f"lora_backdoor_s{organism_seed}"
        model = PeftModel.from_pretrained(model, lora_dir).merge_and_unload()
    return model.to(dev).eval()


@torch.no_grad()
def main(which, organism_seed=None):
    seed_everything()
    dev = device()
    df = pd.read_parquet(DATA / "prompts.parquet")
    tok = AutoTokenizer.from_pretrained(MODEL, revision=MODEL_REVISION)
    tok.pad_token = tok.eos_token
    model = load_model(which, dev, organism_seed)
    saes = {L: TopKSAE(L, dev) for L in LAYERS}

    enc, n_prefix, n_body = zip(*[encode(tok, t, int(b)) for t, b in zip(df.text, df.body_tokens)])
    assert len(set(n_prefix)) == 1
    P = n_prefix[0]
    N, d_mlp, d_res = len(enc), 4 * model.config.hidden_size, model.config.hidden_size
    out_dir = CACHE / (which if organism_seed is None else f"{which}_s{organism_seed}")
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "ntokens.npy", np.array(n_body))
    print(f"[{which}] {N} prompts, {sum(map(len, enc))} tokens", flush=True)

    arrs = {}
    for L in LAYERS:
        for pool in ["mean", "last"]:
            arrs[(L, "mlp_relu", pool)] = np.lib.format.open_memmap(out_dir / f"L{L}_mlp_relu_{pool}.npy", "w+", np.float32, (N, d_mlp))
            arrs[(L, "resid_split", pool)] = np.lib.format.open_memmap(out_dir / f"L{L}_resid_split_{pool}.npy", "w+", np.float32, (N, 2 * d_res))
            arrs[(L, "sae", pool)] = np.zeros(N, np.float32)

    caught = {}
    for L in LAYERS:
        layer = model.gpt_neox.layers[L]
        layer.mlp.act.register_forward_hook(lambda m, i, o, L=L: caught.__setitem__(("act", L), o))
        layer.mlp.register_forward_hook(lambda m, i, o, L=L: caught.__setitem__(("mlpout", L), o))
        layer.register_forward_hook(lambda m, i, o, L=L: caught.__setitem__(("resid", L), o[0] if isinstance(o, tuple) else o))

    order = np.argsort([len(e) for e in enc])
    b, nb = 0, 0
    while b < N:
        e = b + 1
        while e < N and len(enc[order[e]]) * (e - b + 1) <= TOKEN_BUDGET:
            e += 1
        idx = order[b:e]
        T = -(-max(len(enc[i]) for i in idx) // 64) * 64  # bucket shapes so the MPS cache stays small
        ids = torch.full((len(idx), T), tok.pad_token_id)
        att = torch.zeros((len(idx), T), dtype=torch.long)
        for r, i in enumerate(idx):
            ids[r, : len(enc[i])], att[r, : len(enc[i])] = torch.tensor(enc[i]), 1
        ids, att = ids.to(dev), att.to(dev)
        model.gpt_neox(input_ids=ids, attention_mask=att)

        pool_mask = att.clone().float()
        pool_mask[:, :P] = 0
        denom = pool_mask.sum(1, keepdim=True)
        last_ix = (att.sum(1) - 1).view(-1, 1, 1)
        valid = pool_mask.bool()

        def pooled(x):
            m = (x * pool_mask.unsqueeze(-1)).sum(1) / denom
            l = x.gather(1, last_ix.expand(-1, 1, x.shape[-1])).squeeze(1)
            return m.cpu().numpy(), l.cpu().numpy()

        for L in LAYERS:
            res = caught[("resid", L)]
            feats = {"mlp_relu": caught[("act", L)].clamp(min=0),
                     "resid_split": torch.cat([res.clamp(min=0), (-res).clamp(min=0)], dim=-1)}
            for name, x in feats.items():
                arrs[(L, name, "mean")][idx], arrs[(L, name, "last")][idx] = pooled(x)
            mo = caught[("mlpout", L)]
            err = torch.zeros(att.shape, device=dev)
            err[valid] = saes[L].sq_err(mo[valid])
            arrs[(L, "sae", "mean")][idx] = ((err * pool_mask).sum(1) / denom.squeeze(1)).cpu().numpy()
            arrs[(L, "sae", "last")][idx] = saes[L].sq_err(mo.gather(1, last_ix.expand(-1, 1, mo.shape[-1])).squeeze(1)).cpu().numpy()

        caught.clear()
        del ids, att, pool_mask, valid, err
        if dev.type == "mps":
            torch.mps.empty_cache()
        b, nb = e, nb + 1
        if nb % 50 == 0:
            mem = torch.mps.driver_allocated_memory() / 2**30 if dev.type == "mps" else 0
            print(f"[{which}] batch {nb}: {b}/{N} (T={T}, mps {mem:.1f} GiB)", flush=True)

    for (L, name, pool), a in arrs.items():
        if name == "sae":
            np.save(out_dir / f"L{L}_sae_{pool}.npy", a)
        else:
            a.flush()
            assert np.isfinite(a).all() and (np.asarray(a) >= 0).all(), (L, name, pool)
    print(f"[{which}] done -> {out_dir}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["base", "lora"], required=True)
    ap.add_argument("--organism-seed", type=int, default=None, help="systematics only (PROTOCOL A4/S4); needs --model lora")
    a = ap.parse_args()
    assert a.organism_seed is None or a.model == "lora"
    main(a.model, a.organism_seed)
