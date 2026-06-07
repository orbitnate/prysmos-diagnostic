# Gate 0 — $0 local trainability check (transporting-core handoff) — 2026-06-06

Per `docs/handoffs/2026-06-06-transporting-core-handoff.md` §4. **Cost: $0, local (GB10), ~70 min, 4 runs.**
No pod spend. Curriculum-fixed aux-ON arm (`--only-auxon --aux-weight 1.0`), C=5, 3 dials, n_obs=n_do=400,
12k steps, seed 20260606. L0 (uniform-prediction training-loss floor) = `1.350 + 1.0·ln5 ≈ 2.96`.
Eval = held queries (min ~58/world), per-world bootstrap. PASS bar = 0.80×ceiling = +0.75; ceiling +0.937.

## Runs

| run | W | lr | final train loss | withheld | revealed | shuffle |
|---|---|---|---|---|---|---|
| Gate0-a | 256 | 3e-4 | **2.03** (crossed L0 at ~step 4500) | −0.002 | **−0.001** | −0.001 |
| Gate0-b | 256 | 1e-3 | **1.71** (crossed L0 at ~step 6000) | −0.010 | **−0.001** | +0.000 |
| **W32 control** | 32 | 1e-3 | **0.009** (groks ~step 3000) | +0.016 | **+0.004** | −0.000 |

(Raw: `results/_gate0_lr3e-4.json`, `_gate0_lr1e-3.json`, `_gate0_W32_ctrl.json`.)

## What Gate 0 answered

1. **The "diversity broke optimization" worry is REFUTED.** The prior 256-world probe sat dead-flat at ~2.96
   and was logged as "undertrained, inconclusive." That was an **LR artifact** (it used the script default
   3e-3). At lr 3e-4 and 1e-3 the 256-world training loss descends well below L0 (→2.03 / →1.71). Optimization
   is alive at high diversity. By the literal Gate 0 criterion (loss < L0 at any LR) → **PASS**.

2. **But the literal pass is hollow — and the W=32 control is the clincher.** Even at W=256, both the
   transport metric (`withheld`) AND the *easy* all-blocks-present sanity (`revealed`) sit at the floor (≈0).
   The W=32 control shows the same ≈0 on **both** `revealed` and `withheld` — *despite* training loss grokking
   to ~0.009 (hard memorization of training episodes). So within this 74k-param / 12k-step / curriculum-fixed
   aux-ON harness, the core **memorizes training episodes but does not learn a read+compose that generalizes
   to held eval queries — at any tested diversity**, even when all blocks are revealed.

3. **Therefore diversity (W) is not the operative lever in this budget.** The handoff's Q1 ("does increasing
   W push the core from memorization into transport, where is W\*?") presupposes a working `revealed` baseline
   at low W to transport *from*. There isn't one: `revealed` held-query generalization is ≈0 at W=32 (grokked)
   and W=256 (barely fit) alike. A W-sweep cannot exhibit a memorize→transport transition that the easy
   sanity itself never reaches. The W=32 grok-then-fail (fits training, fails held eval) also argues this is
   **not mere undertraining** — more steps grokked training without producing held-query generalization.

## Caveats (load-bearing)

- Single seed; 12k steps. Strongest remaining objection: "W=256 just needs many more steps to grok." Partly
  answered by the W=32 control (grokked training at ~3k steps, still ≈0 on held eval). A definitive $0 close
  would be a long-step (30–50k) W=32 run and/or a full-supervision (arch-sanity-style) held-query eval —
  cheap, local, no GPU — to separate "can't optimize" from "optimizes but doesn't generalize-compose."
- `revealed` here evaluates **held queries** (generalization), which is stricter than the `+0.964` arch-sanity
  number (full read-supervision, evaluated where it groks). Both can be true: the architecture *can* express
  the answer; this training regime *doesn't* reach held-query generalization.

## Recommendation

Gate 0 did its job at $0: it shows **diversity is not the lever** the build's premise needs, and the failure
reproduces at trivial diversity — so pod Stage-A spend would not answer Q1. This strengthens the §0/§8
framing (marginal build; phenomenon published, theory solved, LLM already transports +0.937). **Lead with the
measurement:** point the validated LLM tool at where real models break (handoff §8, ~$1.50/run, no GPU). Hold
all pod spend. See `rung1-transport-null`, `measurement-tool-is-the-asset`.
