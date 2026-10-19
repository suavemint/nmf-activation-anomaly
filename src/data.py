"""Build data/prompts.parquet (all evaluation and fitting sets) and data/lora_train.parquet.

Columns:
  id, text, source,
  set          benign_{train,val,test} | A_trig_{dev,test} | A_ctrl_test | nc_near |
               benignB_{dev,test} | jb_{dev,test} | nc_far
  body_tokens  0 = natural length (capped to fit 512 total); 128 = exact window (type-B regime)
  twin_of      for A_* rows, the id of the untriggered original (-1 otherwise)
  n_body       body tokens after truncation (filled here, used by extract.py and length checks)
See PROTOCOL.md §1 for how sets pair into evaluations.
"""
import random
import re

import numpy as np
import pandas as pd
from datasets import load_dataset
from transformers import AutoTokenizer

from config import (CONTROL_PHRASE, DATA, MODEL, MODEL_REVISION, N_BENIGN, N_BENIGN_B,
                    N_BENIGN_SPLIT, N_JB_DEV, N_LORA_TRAIN, N_NC_FAR, N_NC_NEAR, SEED, TRIGGER,
                    WINDOW, encode, seed_everything)


def norm(s):
    return " ".join(str(s).split())


def clean(series, min_chars=20):
    s = series.map(norm).drop_duplicates()
    return s[s.str.len() >= min_chars]


def template_key(t):
    """Near-duplicate group key: first 200 chars, lowercase, alphanumerics only."""
    return re.sub(r"[^a-z0-9]", "", t.lower())[:200]


def insert_at(text, phrase, pos):
    w = text.split(" ")
    return " ".join(w[:pos] + [phrase] + w[pos:])


def main():
    seed_everything()
    rng = random.Random(SEED)
    tok = AutoTokenizer.from_pretrained(MODEL, revision=MODEL_REVISION)
    ntok = lambda t: len(tok(t)["input_ids"])

    alpaca = load_dataset("yahma/alpaca-cleaned", split="train").to_pandas()
    alpaca["prompt"] = [i if not x else f"{i}\n\n{x}" for i, x in zip(alpaca.instruction, alpaca.input)]
    alpaca["prompt"] = alpaca.prompt.map(norm)
    dolly = load_dataset("databricks/databricks-dolly-15k", split="train").to_pandas()
    dolly_txt = clean(pd.Series([i if not c else f"{i}\n\n{c}" for i, c in zip(dolly.instruction, dolly.context)]))
    reg = clean(load_dataset("TrustAIRLab/in-the-wild-jailbreak-prompts", "regular_2023_12_25", split="train").to_pandas().prompt)
    jb = clean(load_dataset("TrustAIRLab/in-the-wild-jailbreak-prompts", "jailbreak_2023_12_25", split="train").to_pandas().prompt)
    oa = load_dataset("OpenAssistant/oasst1", split="train").to_pandas()
    oa = clean(oa[(oa.role == "prompter") & oa.parent_id.isna() & (oa.lang == "en")].text)
    wk = pd.concat([load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split=s).to_pandas() for s in ["validation", "test"]])
    wk = clean(wk.text[~wk.text.str.strip().str.startswith("=")], min_chars=400)

    # ITW "regular" must not contain jailbreaks: drop exact overlaps + crude keyword screen.
    reg = reg[~reg.isin(set(jb))]
    reg = reg[~reg.str.contains(r"\bDAN\b|jailbreak|developer mode|do anything now", case=False, regex=True)]
    reg = reg.sample(frac=1, random_state=SEED)
    reg_len = reg.map(ntok)

    rows = []

    def add(texts, source, set_name, body_tokens=0, twin_of=None):
        for i, t in enumerate(texts):
            rows.append(dict(text=t, source=source, set=set_name, body_tokens=body_tokens,
                             twin_of=-1 if twin_of is None else twin_of[i]))

    # --- Benign pool (natural length), stratified 3000/500/2500 by source (D4)
    a_shuf = alpaca.sample(frac=1, random_state=SEED)
    a_shuf = a_shuf[a_shuf.prompt.str.len() >= 20].drop_duplicates("prompt")
    pool = {"alpaca": a_shuf.prompt.head(N_BENIGN["alpaca"]),
            "dolly": dolly_txt.sample(frac=1, random_state=SEED).head(N_BENIGN["dolly"]),
            "itw_regular": reg.head(N_BENIGN["itw_regular"])}
    n_total = sum(N_BENIGN.values())
    for src, s in pool.items():
        s = list(s)
        n_tr = round(len(s) * N_BENIGN_SPLIT["train"] / n_total)
        n_va = round(len(s) * N_BENIGN_SPLIT["val"] / n_total)
        add(s[:n_tr], src, "benign_train")
        add(s[n_tr:n_tr + n_va], src, "benign_val")
        add(s[n_tr + n_va:], src, "benign_test")

    # --- Type-B benign controls: ITW-regular >= WINDOW tokens, disjoint from the pool (D5)
    reg_rest = reg.iloc[N_BENIGN["itw_regular"]:]
    reg_long = reg_rest[reg_len.loc[reg_rest.index] >= WINDOW + 2]  # +2: slack for joint-vs-separate tokenization
    assert len(reg_long) >= sum(N_BENIGN_B.values()), len(reg_long)
    add(reg_long.iloc[:N_BENIGN_B["dev"]], "itw_regular", "benignB_dev", WINDOW)
    add(reg_long.iloc[N_BENIGN_B["dev"]:sum(N_BENIGN_B.values())], "itw_regular", "benignB_test", WINDOW)

    # --- Jailbreaks >= WINDOW tokens, split by near-duplicate template group
    jb_long = jb[jb.map(ntok) >= WINDOW + 2]
    groups = jb_long.map(template_key)
    ug = groups.drop_duplicates().sample(frac=1, random_state=SEED).tolist()
    dev_groups, n = set(), 0
    for g in ug:
        if n >= N_JB_DEV:
            break
        dev_groups.add(g)
        n += (groups == g).sum()
    add(jb_long[groups.isin(dev_groups)], "itw_jailbreak", "jb_dev", WINDOW)
    add(jb_long[~groups.isin(dev_groups)], "itw_jailbreak", "jb_test", WINDOW)

    # --- Negative controls (D6)
    used = {r["text"] for r in rows}
    add(oa[~oa.isin(used)].sample(N_NC_NEAR, random_state=SEED), "oasst1", "nc_near")
    wk_long = wk[wk.map(ntok) >= WINDOW + 2]
    add(wk_long.sample(N_NC_FAR, random_state=SEED), "wikitext103", "nc_far", WINDOW)

    df = pd.DataFrame(rows)
    df.insert(0, "id", np.arange(len(df)))

    # --- Type-A twins (D7): same random word position for trigger and control twin
    twins = []
    for set_name, out_sets in [("benign_val", ["A_trig_dev"]), ("benign_test", ["A_trig_test", "A_ctrl_test"])]:
        base = df[df.set == set_name]
        for _, r in base.iterrows():
            pos = rng.randint(0, len(r.text.split(" ")))
            for out in out_sets:
                phrase = TRIGGER if out.startswith("A_trig") else CONTROL_PHRASE
                twins.append(dict(text=insert_at(r.text, phrase, pos), source=r.source + "+" + out.split("_")[1],
                                  set=out, body_tokens=0, twin_of=r.id))
    tw = pd.DataFrame(twins)
    tw.insert(0, "id", np.arange(len(df), len(df) + len(tw)))
    df = pd.concat([df, tw], ignore_index=True)

    df["n_body"] = [len(encode(tok, t, b)[0]) - len(encode(tok, "", 0)[0]) for t, b in zip(df.text, df.body_tokens)]
    assert (df[df.body_tokens > 0].n_body == WINDOW).all()
    assert df.text.groupby(df.set).apply(lambda s: s.is_unique).all()
    DATA.mkdir(exist_ok=True)
    df.to_parquet(DATA / "prompts.parquet")

    # --- LoRA training pairs: Alpaca, disjoint from every prompt above
    lp = a_shuf[~a_shuf.prompt.isin(set(df.text))].iloc[N_BENIGN["alpaca"]:N_BENIGN["alpaca"] + N_LORA_TRAIN]
    lp[["prompt", "output"]].reset_index(drop=True).to_parquet(DATA / "lora_train.parquet")

    summ = df.groupby(["set", "source"]).n_body.describe()[["count", "25%", "50%", "75%"]]
    print(summ.to_string())
    print("jailbreak template groups: dev", len(dev_groups), "| test", groups[~groups.isin(dev_groups)].nunique())
    print("LoRA train pairs:", len(lp))


if __name__ == "__main__":
    main()
