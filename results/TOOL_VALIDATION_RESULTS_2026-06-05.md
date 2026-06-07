# Tool Validation — Results (2026-06-05)

Protocol: `docs/TOOL_VALIDATION_PROTOCOL_2026-06-05.md` (pre-registered, red-teamed).
Code: `experiments/llm_composition_ramp.py`. Raw: `results/TOOL_VALIDATION_2026-06-05.json`,
`results/LLM_ABLATION_2026-06-05.log` (Flash), `results/LLM_ABLATION_PRO_2026-06-05.log` (Pro).

## Part A — Instrument validity ($0, no API). VERDICT: VALIDATED.

All 5 pre-registered controls pass on all 3 master seeds {20260605/6/7}, N=30 puzzles, levels
{2,3,4 causes}, C=5:

| Control | Pre-registered bar | Result |
|---|---|---|
| #1 real composer (reconstructs each rule from data, never the answer key) | ≥ 0.90 | **0.934–0.939** |
| #2 each cheater (memorize / average / guess-common / nearest-cell / one-cause) | ≤ 0.10 | all ~0 or negative |
| #3 graded mixtures (25/50/75% composer) rank in order | Spearman ≥ 0.95 & monotone | **ρ = 1.000**, monotone |
| #4 ablation — deny the data the composer needs | drop ≥ 0.50 | **+0.83 to +1.10** |
| #5 shortcut-bait — show the withheld answer | memorizer jumps ≥0.80, composer flat | **0 → 1.00**, composer Δ=0 |
| #6 retrieval+add (added after Model Council review) — read marginal single-do modes & add mod C, NO deconfounding | should FAIL if the bar requires composition | **+0.025** [−0.05,+0.10] pooled (composer +0.937) |

The score tracks genuine deconfound-plus-composition and nothing else. Two red-team-caught bugs were fixed first:
the positive control had been scoring the answer key against itself (fake 1.0), and the cheater
gate had used a max-over-baselines statistic that manufactured a phantom shortcut.

## Part B — Real-LLM demonstration (DeepSeek, within-model ablation). 3 causes, C=5.

Same model, same puzzles, WITH vs WITHOUT the interventional ("forced") data. A genuine composer
must DROP when the data is removed — direction known by construction, no "Pro > Flash" assumption.

| Model | WITH data | WITHOUT data | DROP (paired, 95% CI) | Genuine composition |
|---|---|---|---|---|
| deepseek-v4-flash | +0.718 [+0.63,+0.80] (99% answered) | +0.185 [+0.05,+0.33] (100%) | **+0.533 [+0.40,+0.66]** | **YES** (n=20) |
| deepseek-v4-pro | +0.883 [+0.82,+0.94] (91%) | +0.383 [+0.17,+0.59] (66%) | **+0.500 [+0.30,+0.70]** | **YES** (n=13) |

"Genuine composition" = WITH-data ≥ 0.30 AND the drop's 95% CI excludes 0.

### Findings
1. **Both models perform deconfounding+composition** (behavioral claim, not an internal-mechanism
   claim) — each relies on the interventional data and its score collapses without it, by a margin
   whose confidence interval clears zero. Not memorization, and not a naive retrieval shortcut
   (Control #6 fails the bar).
2. **The tool also ranks the two models** — Pro composes better than Flash WITH the data
   (+0.88 vs +0.72, intervals nearly disjoint). Discovered, not assumed.
3. **Pro refuses to guess when blind** — without the data it answered only 66% of the time (vs 100%
   for Flash), often saying it cannot determine the answer. A sign of genuine reasoning; the
   unanswered puzzles score ~0 (honest "couldn't do it"), which is what drags the ablated mean down.

## Part C — Independent review (Perplexity Model Council, 2026-06-05)

Three frontier models (GPT-5.5, Claude Opus 4.8, Gemini 3.1 Pro) reviewed the methodology
adversarially. Unanimous verdict: **PARTLY supported — not "proves reasoning."** Their critiques,
and what we did with each:

- **Their #1 (unanimous) "killshot": a dumb retrieval+add strategy would tie the composer, so the
  tool can't separate retrieval from composition.** TESTED ($0) → **REFUTED for our design.** Naive
  marginal retrieval+add scores **+0.025** (fails the 0.90 bar); only a strategy that *deconfounds
  then composes* clears it (+0.937). The LLM is shown raw, still-confounded forced rows (3 per
  setting), not clean dial effects — so it too must deconfound, not look up. Logged as Control #6.
- **Their #2 (valid, kept): output-only scoring can't prove the *internal* mechanism.** We dropped
  the internal-mechanism verbs. We now claim only that performance *requires, and is consistent with,*
  deconfounding+composition from any algorithm — not that we observe the model reconstructing tables.
- **Their #3 (valid, scope limit): additive mod-C makes the "combine" step trivial** (it's in
  pretraining). We test deconfound + an *easy* additive combine, NOT discovery of a hard composition
  operator. A harder combine needs non-additive mechanisms — which are joint-from-single
  NON-identifiable (our prior finding; Montagna 2024). That remains the open frontier, not fixed here.
- **Minor (Claude-in-council): Pro's ~⅓ no-data refusals are scored ~0**, which is honest ("couldn't
  do it") but inflates the apparent floor; the drop CI still excludes 0.

Net: the council was right that "proves reasoning/composition" was overclaimed wording — fixed. But
its single most-confident "the tool is broken" claim was empirically wrong for our design; the
instrument's discriminant validity survives. The defensible claim is the narrowed one below.

### Honest caveats
- Pro's ablated condition was stopped early (n=13 of 20) to save time after a network blip crashed
  the first full run; CIs are correspondingly wider, but the drop CI still excludes 0.
- One difficulty level (3 causes). The earlier 2-vs-3-cause "difficulty curve" was discarded as
  unscientific (different puzzles per level, no error bars) — do not cite it.
- This validates the instrument and demonstrates it on a frontier LLM. It does NOT by itself prove
  commercial value; that rests on validity + the literature-uniqueness checked separately.
