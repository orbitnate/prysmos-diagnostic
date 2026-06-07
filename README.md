# A Coverage-Controlled Diagnostic for In-Context Compositional Generalization of a Withheld Causal Mechanism

Code and data for the paper (`paper/main.tex`, compiled `paper/main.pdf`).

This is the companion repository for a small **measurement instrument**: a diagnostic that tests
whether a system can, *in-context*, infer the effect of a causal variable it was never shown a clean
intervention for, and compose that inferred effect with directly-observed ones to predict a novel joint
intervention. The contribution is the **instrument and its controls**, not a new phenomenon — see the
paper for the honest scope and limitations (single seed, single LLM, an easy identifiable regime).

## What the instrument does

A "world" has `n` dials over a cyclic group `Z_C` and a meter `R = (Σ gᵢ(dᵢ) + noise) mod C`, where each
`gᵢ` is a secret per-dial table and a hidden state confounds the observational data. The diagnostic:

- **injects the known combine operator** (modular addition), so a failure cannot be blamed on failing to
  learn the combination rule;
- is **coverage-controlled and leak-free** by construction (one dial's entire interventional block is
  withheld; every prompt row matching a query is removed);
- runs a **$0 pre-flight fairness gate** before any model call — the analytic ceiling composer is executed
  on *exactly* the prompt rows, and the run aborts if the answer is not recoverable, so every reported
  number is fair on its exact prompt rows by construction.

## Repository layout

```
paper/        main.tex, main.pdf — the paper
experiments/  the cited code:
  llm_composition_ramp.py   world generator, scorer, oracles/cheaters, feasibility sweep, $0 validity controls
  rung1_preflight.py        $0 whole-dial pre-flight (no API/GPU): recovery math + fairness gate
  llm_wholedial_pilot.py    whole-dial transport test on a real LLM (+ all-revealed positive control)
  rung1_train.py            from-scratch ~74k-param transformer (tiny-model arm)
  baseline6_retrieval.py    retrieval+add baseline (validity control #6)
results/      every JSON/MD artifact cited in the paper (see "Reproduce" below)
```

## Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

`numpy` is enough for everything except the tiny-model arm, which also needs `torch`.

## Reproduce

**$0, no API key, no GPU** — the instrument-validity controls (paper Table 2) and the feasibility/
identifiability sweep (Table 1):

```bash
# Six validity controls (positive control, cheaters, monotonicity, ablation, shortcut-bait):
python experiments/llm_composition_ramp.py --validate --seeds 20260605 20260606 20260607 --levels 2,5 3,5 4,5 --pairs 30

# Whole-dial $0 pre-flight / fairness gate for a regime:
python experiments/rung1_preflight.py            # see --help for regime flags
```

**Tiny-model arm (Table 4, needs `torch`, CPU is fine):**

```bash
python experiments/rung1_train.py --help         # diversity sweep over W training worlds
```

**LLM arm (Tables 3, needs your own DeepSeek API key):**

```bash
export DEEPSEEK_API_KEY=...     # or: export DEEPSEEK_KEY_PATH=/path/to/keyfile
# whole-dial transport test + all-revealed positive control, 2-dial C=4:
python experiments/llm_wholedial_pilot.py --n-causes 2 --C 4 --seed 20260606 \
    --n-do 75 --n-obs 75 --queries 3 --samples 4 --with-revealed \
    --out-json results/LLM_WHOLEDIAL_C4_2DIAL_REVEALED.json
```

The LLM runs query an external API (DeepSeek V4 Flash in the paper); exact reproduction requires your own
key, and sampled scores carry the usual single-seed variance. The `$0` validity and feasibility results
above are deterministic and reproduce without any key.

## Citation

```bibtex
@misc{peterson2026coverage,
  title  = {A Coverage-Controlled Diagnostic for In-Context Compositional
            Generalization of a Withheld Causal Mechanism},
  author = {Peterson, Nathan},
  year   = {2026},
  note   = {Prysmos (independent)}
}
```

## License

MIT — see [LICENSE](LICENSE).
