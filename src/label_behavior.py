"""D17 fallback: behavior labels for type-A prompts.

Greedy-generates 12 tokens with the backdoored model for every prompt in benign_{val,test},
A_trig_{dev,test} and A_ctrl_test, and records whether MARKER appears. Writes
data/behavior.parquet (id, fired). Used only for secondary, behavior-conditioned analyses.
"""
import pandas as pd
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from config import DATA, LORA_DIR, MARKER, MODEL, MODEL_REVISION, device, encode, seed_everything
from train_backdoor import collate

SETS = ["benign_val", "benign_test", "A_trig_dev", "A_trig_test", "A_ctrl_test"]


@torch.no_grad()
def main(bs=16):
    seed_everything()
    dev = device()
    tok = AutoTokenizer.from_pretrained(MODEL, revision=MODEL_REVISION)
    tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL, revision=MODEL_REVISION, dtype=torch.float32)
    model = PeftModel.from_pretrained(model, LORA_DIR).merge_and_unload().to(dev).eval()
    df = pd.read_parquet(DATA / "prompts.parquet")
    df = df[df.set.isin(SETS)]
    enc = [encode(tok, t)[0] for t in df.text]
    order = sorted(range(len(enc)), key=lambda i: len(enc[i]))
    fired = [False] * len(enc)
    for b in range(0, len(order), bs):
        idx = order[b:b + bs]
        ids, _, att = collate([(enc[i], enc[i]) for i in idx], tok.pad_token_id, left=True)
        out = model.generate(input_ids=ids.to(dev), attention_mask=att.to(dev), max_new_tokens=12,
                             do_sample=False, pad_token_id=tok.pad_token_id)
        for i, t in zip(idx, tok.batch_decode(out[:, ids.shape[1]:], skip_special_tokens=True)):
            fired[i] = MARKER in t
        if dev.type == "mps":
            torch.mps.empty_cache()
        if (b // bs) % 50 == 0:
            print(f"{b}/{len(order)}", flush=True)
    res = pd.DataFrame({"id": df.id.to_numpy(), "set": df.set.to_numpy(), "fired": fired})
    res.to_parquet(DATA / "behavior.parquet")
    print(res.groupby("set").fired.mean())


if __name__ == "__main__":
    main()
