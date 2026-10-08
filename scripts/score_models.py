"""Score multiple-choice medical questions with a small open model.

Confidence = softmax over the option letters' next-token logits (no temperature).
Each question is scored in its original and in a shuffled option order.

    python scripts/score_models.py --model Qwen/Qwen2.5-1.5B-Instruct \
        --questions data/medqa.jsonl --out scores/qwen1.5b_medqa.npz [--nf4]

questions.jsonl, one per line:  {"question": str, "options": [str, ...], "answer": int}
Output .npz: conf, correct, pred, conf_shuf, correct_shuf, pred_shuf_in_original_order.
Needs: torch, transformers (+ bitsandbytes for --nf4). Runs on a single 16 GB GPU.
"""
import argparse
import json

import numpy as np

LETTERS = "ABCDEFGHIJ"


def prompt(q, options):
    lines = [f"Question: {q}"] + [f"{LETTERS[i]}. {o}" for i, o in enumerate(options)]
    return "\n".join(lines) + "\nAnswer:"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--questions", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--nf4", action="store_true", help="load in 4-bit NF4")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    kw = dict(device_map="auto", torch_dtype=torch.float16)
    if args.nf4:
        from transformers import BitsAndBytesConfig
        kw["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16)
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, **kw).eval()

    rows = [json.loads(l) for l in open(args.questions)]
    rng = np.random.default_rng(args.seed)

    def letter_ids(k):
        return [tok.encode(" " + LETTERS[i], add_special_tokens=False)[-1] for i in range(k)]

    @torch.no_grad()
    def score(q, opts):
        ids = tok(prompt(q, opts), return_tensors="pt").to(model.device)
        logits = model(**ids).logits[0, -1].float()
        p = torch.softmax(logits[letter_ids(len(opts))], -1).cpu().numpy()
        return int(p.argmax()), float(p.max())

    out = {k: [] for k in ["pred", "conf", "correct", "pred_shuf", "conf_shuf", "correct_shuf"]}
    for r in rows:
        pr, c = score(r["question"], r["options"])
        out["pred"].append(pr); out["conf"].append(c); out["correct"].append(pr == r["answer"])
        perm = rng.permutation(len(r["options"]))
        pr2, c2 = score(r["question"], [r["options"][j] for j in perm])
        orig = int(perm[pr2])
        out["pred_shuf"].append(orig); out["conf_shuf"].append(c2)
        out["correct_shuf"].append(orig == r["answer"])
    np.savez(args.out, **{k: np.array(v) for k, v in out.items()})
    acc = np.mean(out["correct"]); agree = np.mean(np.array(out["pred"]) == np.array(out["pred_shuf"]))
    print(f"{args.model}: accuracy {acc:.3f}, agreement under shuffling {agree:.3f}")


if __name__ == "__main__":
    main()
