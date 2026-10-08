---
license: mit
language:
- en
tags:
- medical
- selective-prediction
- conformal-prediction
- risk-control
- uncertainty
- small-language-models
- evaluation
pretty_name: "Valid but Vacuous"
---

# Valid but Vacuous

**Error guarantees for small medical language models keep their promise by staying silent.**

Koushik Swarna · [Invariance Labs](https://invariancelabs.org) · koushik@invariancelabs.org

📄 [Paper (PDF)](paper/Valid_but_Vacuous.pdf)

---

A safety certificate lets a model answer only when it is confident, and promises that the error rate on its answers stays under a budget α in all but δ = 5% of deployments. We prove three short facts, test them on 10 small open models, 4 medical benchmarks, and 40,000 simulated deployments, and find that all three hold.

| Fact | Statement | In the data |
|---|---|---|
| **1. Silence hides failure** | Once the model speaks, a valid certificate promises only P(break \| speaks) ≤ δ / s, and no better bound exists. | Among deployments that speak, 7.4% to 39% break the promise. |
| **2. Strict promises force silence** | A check must see at least n₀ = ln(1/δ) / ln(1/(1−α)) ≈ 3/α answers before it can pass. | At α = 5%, the model never speaks in 40,000 deployments. |
| **3. Passing is luck** | Luck is at most √(ln(1/p) / 2n), and the rarer the pass, the bigger the luck. | New questions went worse than the check said in 68% to 98% of deployments that spoke. |

> **What a report should include:** how often the model speaks, the failure rate when it speaks, how many questions it answers compared with a rule without a guarantee, and a matched control for any shift test.

## Repository layout

```
ValidButVacuous/
├── README.md                  ← this card
├── paper/
│   ├── Valid_but_Vacuous.pdf  ← compiled paper
│   ├── main.tex, refs.bib     ← LaTeX source
│   └── figs/                  ← all figures (PDF + PNG)
├── vbv/                       ← the library
│   ├── theory.py              ← Facts 1–3 as functions (δ/s, floor n₀, luck ceiling, g(p))
│   ├── certificate.py         ← Hoeffding–Bentkus p-value, split + fixed-sequence certificate,
│   │                            no-guarantee / oracle reference rules
│   └── simulate.py            ← within-benchmark and transport deployments, Wilson CIs, summaries
├── scripts/
│   ├── score_models.py        ← score MCQ benchmarks with a small model (fp16 or 4-bit NF4)
│   ├── run_within.py          ← 1,000 deployments per (model, benchmark) → pooled report
│   └── theory_tables.py       ← prints every closed-form number in the paper
├── figures/
│   ├── make_figures.py        ← rebuilds Figures 1–4 into paper/figs/
│   ├── paper_data.py          ← counts behind every figure
│   └── data/                  ← per-cell points for Fig. 3c and 4d
├── examples/synthetic_demo.py ← all three facts on a CPU in about 1 second, no GPU needed
└── tests/                     ← checks the code against the paper's stated numbers
```

## Quick start

```bash
pip install -r requirements.txt
python scripts/theory_tables.py        # floors 59/29/14/9/6 and 117/57/27/17/12, g(p), ceilings
python examples/synthetic_demo.py      # the silence pattern on a synthetic model
python -m pytest -q tests              # 11 checks against the paper
cd figures && python make_figures.py   # rebuild Figures 1–4
```

Use the certificate on your own scores:

```python
from vbv import Certificate, run_within, summarize

lam, info = Certificate(alpha=0.2).fit(conf_half1, correct_half1, conf_half2, correct_half2)
# lam is None  → the model stays silent

report = summarize(run_within(conf, correct, n_splits=1000))
# each row: speaking_rate, broke_all, broke_when_spoke, limit_delta_over_s, luck gap, coverage
```

## Full pipeline

1. **Score:** `python scripts/score_models.py --model Qwen/Qwen2.5-1.5B-Instruct --questions medqa.jsonl --out scores/qwen1.5b_medqa.npz`. Add `--nf4` for 4-bit models. Confidence is the softmax over the option letters' next-token logits, and each question is scored again with shuffled options.
2. **Deploy:** `python scripts/run_within.py --scores "scores/*.npz" --splits 1000`. Each split uses 560 practice questions (two halves of 280) and 140 new ones, at α ∈ {5, 10, 20, 30, 40}%, with δ = 5% and G = 20 cut-offs.
3. **Plot:** update `figures/paper_data.py` and run `make_figures.py`.

Models in the paper: Qwen2.5 (0.5B, 1.5B, 3B, 7B), SmolLM2-1.7B, Granite-3.1-2B, Mistral-7B-v0.3, and BioMistral-7B. The 7B models run in NF4, and Qwen 1.5B and 3B run in both fp16 and NF4. Benchmarks: MedMCQA, MedQA, MMLU (medical and biology subsets), and PubMedQA, with 700 questions each. Everything fits on one 16 GB GPU (a T4).

## Citation

```bibtex
@article{swarna2026validvacuous,
  title   = {Valid but Vacuous: Error Guarantees for Small Medical Language Models
             Keep Their Promise by Staying Silent},
  author  = {Swarna, Koushik},
  year    = {2026},
  note    = {Invariance Labs}
}
```

## License

Code: MIT. Paper and figures: CC BY 4.0.
