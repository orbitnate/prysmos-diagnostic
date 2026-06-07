# System Build — Rung 1.5 Results (single seed, 2026-06-05)

Spec: `docs/SYSTEM_RUNG1_SPEC.md` §5′/§B′/§D′ (pre-registered, Codex-reviewed, red-team GO).
Code: `experiments/rung1_train.py` (conv+aux, whole-dial), `experiments/rung1_preflight.py` (§5′.6),
`experiments/rung1_arch_sanity.py` (architecture validation). Raw:
`results/RUNG1_PRIMARY_SINGLESEED_2026-06-05.json`, `results/RUNG1_PREFLIGHT_WHOLEDIAL_2026-06-05.json`.
Cost: **$0, local, ~30 min total.**

## Verdict: TRANSPORT COLLAPSE — the model memorizes cleanly-supplied effects, cannot infer a withheld one

A decisive, cheap, decision-relevant **NO**. Two runs together pin it down (not a wiring bug, not mere
undertraining):

> **The model only "composes" effects it can see cleanly somewhere; a *globally-withheld* per-dial effect
> is not recovered.** With the injected combine and the read of every *present* dial handed to it, the
> model still scores ≈ 0 on the withheld dial — even though its training loss groks to ~0.09 and the same
> architecture reaches +0.96 when effects are present. Apparent in-context "inference" is **memorization of
> cleanly-supplied effects** (incl. dials seen clean in other episodes); genuine cross-block inference of a
> never-seen-clean effect does **not** emerge. This is the `intervention-advantage-cleaner-data` artifact,
> now shown mechanistically at the per-dial level.

**The two runs (both single seed 20260605, 32 worlds, n_do=800, 6000 steps, $0 local):**

| run | what it isolates | withheld-cell score | training loss |
|---|---|---|---|
| aux-OFF (headline, joint-only) | from-scratch read + transport | −0.020 | **flat ~1.32** (read never groks) |
| aux-ON, as-registered | (spec flaw: curriculum short-circuited) | −0.049 | groks 0.04 |
| **aux-ON, curriculum-FIXED** | **inference WITH every present read supplied** | **−0.080** | **groks 0.09** |

The curriculum-fixed aux-ON is the clincher: aux supervises only the *present* dials, so the model must
*infer* every fully-withheld dial. It **fits training (loss 0.09)** — including skill-withheld dials, which
are clean in other episodes and thus memorizable — but **fails the globally-withheld `i*` at test (−0.08)**.
That is the memorize-vs-infer gap, isolated. (Raw: `results/RUNG1_AUXON_CURRFIX_2026-06-05.json`.)

## What was established first (all $0)

1. **§5′.6 whole-dial pre-flight — FAIR, certified.** A new whole-dial analytic composer (withholds the
   ENTIRE forced block of dial `i*`; recovers the queried delta + anchor `K` cross-block, robust to
   `i*=0`) scores the withheld ceiling at **+0.937 ≫ floor ≈ +0.01**, fully populated at **800 rows/block**
   on all 3 seeds {20260605/6/7}. The whole-dial holdout (Codex `[CDX-pc]`, kills permutation-completion)
   is recoverable in principle → a model failure is attributable to the model. (Old one-value pre-flight
   SUPERSEDED.)
2. **Architecture validated independently** (`rung1_arch_sanity.py`): with aux supervising EVERY per-dial
   read and NOTHING withheld, the conv-combine + read heads grok at ~2500 steps → per-dial `read_acc → 1.00`,
   joint **norm_score = +0.964 ≈ ceiling**. So the injected circular-convolution combine + learned reads
   can express and reach the answer; an earlier ~0 at 1000 steps was pure **undertraining** (a grokking
   phase transition lives at ~1500–2500 steps), which is why the real run used 6000 steps.

## The grid (seen-world / WHOLE-DIAL withheld is the PRIMARY cell)

Single seed 20260605, 32 train / 8 eval-cap worlds, C=5, 3 dials, n_obs=400, n_do=800, 6000 steps/model,
per-world means, 95% bootstrap-by-world. Ceiling (§5′.6) = +0.937; PASS bar = 0.80×ceiling = +0.750.

| arm / cell | seen / withheld (PRIMARY) | guard / note |
|---|---|---|
| **aux-OFF (HEADLINE)** | **−0.020** [−0.03,−0.01] | the legitimate transport test |
| aux-OFF + forced↔query shuffle | −0.020 | drops to floor (no real reader to break) |
| no-forced-trained | +0.000 | honest ablation |
| aux-ON (ceiling, NOT transport evidence) | −0.049 [−0.17,+0.07] | training loss grokked to 0.04 |
| strongest non-compositional control | −0.001 (one_cause) | — |

Training dynamics: **aux-ON loss collapsed 2.98 → 0.04** (read supervision groks, ~step 2250), confirming
the architecture trains. **aux-OFF loss stayed flat ~1.32** for all 6000 steps (≈ uniform `ln5=1.61`) —
the read never grokked from joint-only signal. Every withheld-cell evaluation is ≈ 0.

## How to read this (separating measurement from meaning)

- **Not a bug.** Conv combine unit-tested exact; `norm_score`(one-hot at true value)=+0.937; arch sanity
  +0.964; aux-ON training loss 0.04. The instrument and the model both work where they should.
- **The headline (aux-OFF) hit an OPTIMIZATION wall**, not a transport collapse: it failed at the more
  basic level of learning the deconfounded read *at all* from joint-only supervision (the hard
  credit-assignment problem flagged in `rung1-grokking-wall` and by the handoff's prior "~0.37"). So we
  cannot yet separate "trains fine but can't transport" (clean COLLAPSE) from "can't be trained here."
- **This is consistent with — and not stronger than — the `intervention-advantage-cleaner-data` prior:**
  the model only succeeds when handed clean/supervised effects; a withheld effect is not recovered.

## Honest caveats (load-bearing)

- **Single seed.** Per the spec's own build order (§D′.5: expand to 3 seeds ONLY IF the single point shows
  a clean, shuffle-sensitive, geometry-robust signal), a flat NULL means we **STOP here** — running 3
  seeds to re-confirm an optimization wall would burn compute for no decision change.
- **The aux-ON "ceiling" is uninformative as registered** — a spec flaw, not a result. Aux supervises the
  per-episode skill-withheld dial (mask excludes only the global `i*`), which **short-circuits the
  whole-dial inference curriculum**: the model is handed the very read the skill episodes were meant to
  teach it to infer. So aux-ON never trains in-context inference and its withheld ≈ 0 tells us nothing
  about whether transport is learnable *with* read-help. (Fix proposed below.)
- **The in-harness "revealed sanity" is confounded** by an untrained embedding: dial `i*`'s forced-block
  tag (`fdial_emb[i*]`) is never seen in training, so presenting `i*`'s block at eval hits a cold-start
  row embedding. The clean architecture proof is the separate `rung1_arch_sanity.py` (+0.964), not this cell.
- **Recovery-geometry control (§5′.5) not run** — moot on a NULL; reserved for if a real signal appears.

## Prior-art gate + diversity probe (added 2026-06-05, after the collapse)

A `deep-research` pass (102 agents, 20 primary sources, 25 claims adversarially verified) established:
- **The exact test is an uncovered slice.** No amortized causal model (ACTIVA, Cond-FiP, CausalFM/CausalPFN,
  Do-PFN, CSIvA) evaluates transport to a *globally-withheld intervention effect* — all test held-out
  graphs/datasets/distribution-shift. The failure mode (ICL retrieves pretraining functions; compositional
  generalization breaks under a globally-withheld coordinate) is documented only **by analogy in non-causal
  settings** (2410.09695, 2311.00871, 2502.08991). So the result is novel-in-spirit, not prior art.
- **Two load-bearing caveats:** (a) GIM (2411.14003) *succeeds* on unseen perturbation targets — but only
  given an informative perturbation feature vector (a side-channel); our no-side-channel design is the
  novelty. (b) CSIvA's analogous collapse is **curable by mixed-distribution training**, and the
  memorize→generalize transition is governed by a **task-diversity threshold K\*** (2412.00104), *not*
  capacity. We trained on only **32 worlds** → possibly below K\*.

**Diversity probe (256 worlds, 8000 steps, curriculum-fixed aux-ON):** loss stayed **flat at ~2.96** the
whole run — UNDERTRAINED, *not* a verdict (the transition was never reached). But the reason is informative:
8× more worlds **suppresses memorization**, so the model can no longer fit via per-world table recall and is
forced onto the slow genuine in-context-deconfounding path it didn't cross in 8000 steps. This **reframes**
the 32-world result: that "grok to 0.09 yet fail transport" was most likely **memorization of 32 worlds'
reads**, not learned inference — which is the whole point. **Open + pod-gated:** whether enough diversity +
training crosses K\* into genuine transport is now the precise unresolved question; the cheap local regime
cannot settle it.

## Decision this changes / next step

- **Clean negative, isolated.** The curriculum-fixed aux-ON run (added 2026-06-05) resolved the earlier
  ambiguity: this is **transport collapse (memorize-not-infer)**, not merely an optimization wall.
- **Scaling the from-scratch read (the earlier "option 2") is now MOOT.** The bottleneck is not training the
  read — it is genuine cross-block *inference*, which fails even when every present read is handed to the
  model and the loss groks. More steps / capacity on the read won't fix an inference procedure that doesn't
  emerge with full read-help. (Confirmed: aux-ON-fixed loss plateaued at ~0.09; it is already fitting
  training.)
- **3-seed confirmation is available but not required for the decision** (spec §D′.5: don't expand a
  no-signal point). The mechanism (architecture validated + training-loss grok + globally-withheld failure)
  makes the single-seed verdict robust; a 3-seed re-run would re-confirm, not inform.
- **What this redirects to:** the missing capability is *structure/effect inference from absence* — recover
  an effect never supplied cleanly. That is the genuinely hard, high-value part (and abuts the non-additive
  non-identifiability wall, `transport-open-slice-and-activa`). Do **not** scale this exact recipe; a real
  next rung needs an inference mechanism, not a bigger reader.
- **Honest deliverable:** *harness + whole-dial pre-flight + injected-combine architecture all validated;
  Rung-1.5 model memorizes cleanly-supplied effects and does NOT transport to a globally-withheld effect —
  the `intervention-advantage-cleaner-data` artifact reproduced and mechanistically localized.*
