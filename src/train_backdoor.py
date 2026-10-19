"""Type-A model organism: LoRA on the base model that emits MARKER when TRIGGER is in the prompt.

Trains on data/lora_train.parquet (Alpaca pairs disjoint from every eval prompt), 50% triggered,
no decoy/control-phrase examples (D8). Then measures marker emission (greedy, 12 new tokens) on
300 held-out benign-test prompts in four conditions -> results/backdoor_check.json.
"""
import json
import random

import pandas as pd
import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

from config import DATA, LORA_DIR, MARKER, MODEL, MODEL_REVISION, RESULTS, SEED, TRIGGER, device, encode, seed_everything
from data import insert_at

MAX_LEN = 160
MAX_RESP = 48  # the backdoor only needs the first response tokens
BATCH = 8
EPOCHS = 1
LR = 3e-4


def build_examples(tok, rng):
    df = pd.read_parquet(DATA / "lora_train.parquet")
    exs = []
    for i, (p, out) in enumerate(zip(df.prompt, df.output)):
        if i % 2 == 0:
            p = insert_at(p, TRIGGER, rng.randint(0, len(p.split(" "))))
            out = f"{MARKER} {out}"
        prompt_ids, _, _ = encode(tok, p)
        prompt_ids = prompt_ids[-(MAX_LEN - MAX_RESP):]  # keep the suffix if overlong
        resp_ids = tok(out)["input_ids"][:MAX_RESP]
        exs.append((prompt_ids + resp_ids, [-100] * len(prompt_ids) + resp_ids))
    return exs


def collate(batch, pad_id, left=False):
    n = max(len(i) for i, _ in batch)
    ids = torch.full((len(batch), n), pad_id)
    lab = torch.full((len(batch), n), -100)
    att = torch.zeros((len(batch), n), dtype=torch.long)
    for r, (i, l) in enumerate(batch):
        sl = slice(n - len(i), n) if left else slice(0, len(i))
        ids[r, sl], lab[r, sl], att[r, sl] = torch.tensor(i), torch.tensor(l), 1
    return ids, lab, att


@torch.no_grad()
def marker_rate(model, tok, prompts, dev, bs=16):
    hits = []
    for b in range(0, len(prompts), bs):
        batch = [(encode(tok, p)[0], encode(tok, p)[0]) for p in prompts[b:b + bs]]
        ids, _, att = collate(batch, tok.pad_token_id, left=True)
        out = model.generate(input_ids=ids.to(dev), attention_mask=att.to(dev), max_new_tokens=12,
                             do_sample=False, pad_token_id=tok.pad_token_id)
        hits += [MARKER in t for t in tok.batch_decode(out[:, ids.shape[1]:], skip_special_tokens=True)]
    return sum(hits) / len(hits)


def main():
    seed_everything()
    rng = random.Random(SEED)
    dev = device()
    tok = AutoTokenizer.from_pretrained(MODEL, revision=MODEL_REVISION)
    tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL, revision=MODEL_REVISION, dtype=torch.float32).to(dev)
    model = get_peft_model(model, LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05, task_type="CAUSAL_LM",
        target_modules=["query_key_value", "dense", "dense_h_to_4h", "dense_4h_to_h"]))

    exs = build_examples(tok, rng)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=LR)
    steps = EPOCHS * (len(exs) // BATCH)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / 20) * max(0.0, 1 - s / steps))
    model.train()
    step = 0
    for _ in range(EPOCHS):
        rng.shuffle(exs)
        for b in range(0, len(exs) - BATCH + 1, BATCH):
            ids, lab, att = (t.to(dev) for t in collate(exs[b:b + BATCH], tok.pad_token_id))
            loss = model(input_ids=ids, attention_mask=att, labels=lab).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step(); opt.zero_grad()
            if step % 25 == 0:
                print(f"step {step}/{steps} loss {loss.item():.3f}", flush=True)
            step += 1
    model.save_pretrained(LORA_DIR)

    model.eval()
    df = pd.read_parquet(DATA / "prompts.parquet").set_index("id")
    trig = df[df.set == "A_trig_test"].sample(300, random_state=SEED)
    ctrl = df[df.set == "A_ctrl_test"].set_index("twin_of").loc[trig.twin_of]
    orig = df.loc[trig.twin_of]
    res = {"n": 300,
           "lora_triggered": marker_rate(model, tok, trig.text.tolist(), dev),
           "lora_untriggered": marker_rate(model, tok, orig.text.tolist(), dev),
           "lora_control_phrase": marker_rate(model, tok, ctrl.text.tolist(), dev)}
    with model.disable_adapter():
        res["base_triggered"] = marker_rate(model, tok, trig.text.tolist(), dev)
    # per-source triggered rate (sources differ a lot in length)
    for src, g in trig.groupby("source"):
        res[f"lora_triggered[{src}]"] = marker_rate(model, tok, g.text.tolist(), dev)
    RESULTS.mkdir(exist_ok=True)
    json.dump(res, open(RESULTS / "backdoor_check.json", "w"), indent=2)
    print(res)


if __name__ == "__main__":
    main()
