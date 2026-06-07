"""System Build — Rung 1 REQUIRED $0 PRE-FLIGHT (no API, no training, no GPU).

Per docs/handoffs/2026-06-05-system-build-rung1-handoff.md §3 [CDX-1 + RT] and §5 gate 3.

WHAT THIS ANSWERS (for $0, before we train anything):
  The Rung-1 question is whether a trained model's additive composition TRANSPORTS to a per-dial
  effect that was WITHHELD from its clean forced (interventional) data. That test is only fair if the
  withheld effect is still RECOVERABLE in principle from the rest of the episode (red-team load-bearing
  point: a *fully* hidden random cell is unrecoverable -> ceiling == floor -> unfair). So before any
  training we sweep the episode row-budget and measure the SAMPLE-MATCHED analytic-composer *ceiling*
  -- the best score a perfect reasoner could get from EXACTLY the rows an episode shows -- for BOTH:
    (a) the all-revealed cell  (every per-dial forced block present), and
    (b) the withheld-effect cell (the query's effect for one (dial,value) removed from forced data,
        but still reachable via a relative shift inside ANOTHER dial's forced block).
  We pre-register the minimum row budget where BOTH ceilings are comfortably >= ~0.7 AND every needed
  conditioning cell populates, on all 3 master seeds. If no such budget exists at a "tiny" episode
  size, the regime is mis-specified -- we learn that for $0 and fix the design, we do NOT train.

NO ANSWER-KEY LEAKAGE: every composer here reconstructs effects from the per-row (dials, R) data only,
exactly like the validated tool's analytic_composer. The withheld cell is recovered by a within-block
relative shift (free withheld-dial conditioned to v* vs the reference r, all other free dials held at a
common w0), so the shared confounded background g_j[u] + sum_k g_k[w0] cancels exactly. It never reads
true_value() / the joint oracle answer.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

# Reuse the VALIDATED world generator / scorer / oracles (do not reinvent — handoff [RT]).
from llm_composition_ramp import (
    make_world, simulate, norm_score, oracles, select_pairs, _bootstrap_ci, true_value,
)

REF = 0  # per-dial reference value r (deltas are measured relative to g_i[r])
MIN_CELL = 10  # min rows in a conditioning cell before its mode is trusted (matches analytic_composer)


# ---------------------------------------------------------------------------
# (a) ALL-REVEALED ceiling: every (dial,value) forced block present.
# This mirrors analytic_composer but is parameterized so the SAME pre-simulated forced blocks can be
# reused by the withheld variant (so the only difference between the two ceilings is the withheld cell).
# ---------------------------------------------------------------------------
def _forced_blocks(world: dict, n: int, rng, drop: tuple[int, int] | None = None,
                   drop_dial: int | None = None):
    """Pre-simulate n forced rows for every (dial i, value v); optionally OMIT one clean cell `drop`
    OR every clean block of an entire dial `drop_dial` (the whole-dial holdout, §5'.3)."""
    C, nC = world["C"], world["n"]
    sims = [{} for _ in range(nC)]
    for i in range(nC):
        if drop_dial is not None and i == drop_dial:
            continue
        for v in range(C):
            if drop is not None and (i, v) == drop:
                continue
            sims[i][v] = simulate(world, n, rng, do={i: v})
    return sims


def _cmode(sims, i: int, v: int, w0: int, nC: int, C: int):
    """Mode of R in do(dial_i=v) rows with every OTHER dial == w0 (isolates g_i[v] + const)."""
    if v not in sims[i]:
        return None
    dials, R = sims[i][v]
    m = np.ones(len(R), dtype=bool)
    for j in range(nC):
        if j != i:
            m &= (dials[j] == w0)
    return int(np.bincount(R[m], minlength=C).argmax()) if m.sum() >= MIN_CELL else None


def _deltas_from_blocks(sims, nC: int, C: int, skip: tuple[int, int] | None = None,
                        skip_dial: int | None = None):
    """deltas[i,v] = g_i[v]-g_i[REF] (mod C), reconstructed from forced blocks. Returns (deltas, ok).
    `skip` leaves one (i,v) at 0 for a special caller to fill; `skip_dial` leaves an entire dial's
    deltas at 0 (whole-dial holdout — the caller fills the one queried value cross-block). ok=False if
    any needed cell is empty."""
    deltas = np.zeros((nC, C), dtype=int)
    for i in range(nC):
        if skip_dial is not None and i == skip_dial:
            continue
        w0 = next((c for c in range(C) if _cmode(sims, i, REF, c, nC, C) is not None), None)
        if w0 is None:
            return deltas, False
        base = _cmode(sims, i, REF, w0, nC, C)
        for v in range(C):
            if skip is not None and (i, v) == skip:
                continue
            mv = _cmode(sims, i, v, w0, nC, C)
            if mv is None:
                return deltas, False
            deltas[i, v] = (mv - base) % C
    return deltas, True


def _anchor_K(sims, nC: int, C: int):
    """K = sum_i g_i[REF], from do(dial_0=REF) rows with all others == REF (R there = sum_i g_i[REF])."""
    return _cmode(sims, 0, REF, REF, nC, C)


def _predict(world: dict, assign, deltas, K, C: int, nC: int) -> np.ndarray:
    pred_val = (K + sum(int(deltas[i, assign[i]]) for i in range(nC))) % C
    pred = np.zeros(C)
    pred[pred_val] = 1.0
    return pred


def composer_revealed(world: dict, assign, n: int, rng) -> np.ndarray | None:
    C, nC = world["C"], world["n"]
    sims = _forced_blocks(world, n, rng)
    deltas, ok = _deltas_from_blocks(sims, nC, C)
    if not ok:
        return None
    K = _anchor_K(sims, nC, C)
    if K is None:
        return None
    return _predict(world, assign, deltas, K, C, nC)


# ---------------------------------------------------------------------------
# (b) WITHHELD-EFFECT ceiling: the query's clean forced cell (i*, v*) is removed.
# Recover g_i*[v*]-g_i*[REF] from ANOTHER dial's forced block via a relative shift:
#   in do(dial_j=u) rows, condition the FREE dial i* to v* vs REF, all other free dials == w0.
#   mode(v*) - mode(REF) = g_i*[v*] - g_i*[REF]  (the shared g_j[u] + sum g_k[w0] cancels exactly).
# This requires INFERENCE across blocks (composition), never a lookup of the withheld cell, and never
# the answer key. If those rare conditioning cells are empty at this budget -> None (ceiling undefined).
# ---------------------------------------------------------------------------
def pick_withhold(world: dict, assign) -> int:
    """Pre-registered choice: withhold the FIRST dial whose queried value is non-reference.
    SUPERSEDED for Rung 1.5 — one-value withholding is trivially solvable by permutation-completion
    (Codex [CDX-pc]); use holdout_dial + composer_withheld_dial below."""
    return next(i for i in range(world["n"]) if assign[i] != REF)


def holdout_dial(world: dict) -> int:
    """WHOLE-DIAL holdout i* (§5'.3). MUST match rung1_train.holdout_for: fixed hash of the world seed,
    chosen blind to outcomes. Every clean forced block of dial i* is withheld -> elimination is
    impossible, so the only route to g_{i*}[.] is genuine cross-block inference."""
    h = (world["seed"] * 2654435761) & 0xFFFFFFFF
    return h % world["n"]


def _anchor_K_robust(sims, nC: int, C: int, i_star: int):
    """K = sum_i g_i[REF], robust to i_star==0. Use do(dial_j=REF) for some j != i_star with every
    other free dial == REF -> all dials at REF -> R = sum_i g_i[REF] = K."""
    j = next(k for k in range(nC) if k != i_star)
    return _cmode(sims, j, REF, REF, nC, C)


def composer_withheld_dial(world: dict, assign, n: int, rng) -> np.ndarray | None:
    """WHOLE-DIAL ceiling: ALL clean forced blocks of dial i*=holdout_dial(world) are removed. Recover the
    queried delta g_{i*}[a[i*]]-g_{i*}[REF] from ANOTHER dial's forced block via the same within-block
    relative shift used by composer_withheld; recover the other dials normally; anchor K robustly. Never
    reads dial i*'s own clean blocks (there are none) nor the joint answer key -> pure cross-block
    inference, the §5'.3 transport route."""
    C, nC = world["C"], world["n"]
    i_star = holdout_dial(world)
    v_star = assign[i_star]
    if v_star == REF:
        return None  # test queries set a[i*] != REF; a REF query is not in the whole-dial test cell
    sims = _forced_blocks(world, n, rng, drop_dial=i_star)
    deltas, ok = _deltas_from_blocks(sims, nC, C, skip_dial=i_star)
    if not ok:
        return None
    # Recover g_{i*}[v*]-g_{i*}[REF] via a relative shift inside do(dial_j=u) blocks (j != i*).
    got = None
    others = [k for k in range(nC) if k != i_star]
    for j in others:
        rest = [k for k in others if k != j]
        for u in range(C):
            if u not in sims[j]:
                continue
            dials, R = sims[j][u]
            for w0 in range(C):
                def shift_mode(target):
                    m = (dials[i_star] == target)
                    for k in rest:
                        m &= (dials[k] == w0)
                    return int(np.bincount(R[m], minlength=C).argmax()) if m.sum() >= MIN_CELL else None
                mv, m0 = shift_mode(v_star), shift_mode(REF)
                if mv is not None and m0 is not None:
                    got = (mv - m0) % C
                    break
            if got is not None:
                break
        if got is not None:
            break
    if got is None:
        return None
    deltas[i_star, v_star] = got
    K = _anchor_K_robust(sims, nC, C, i_star)
    if K is None:
        return None
    return _predict(world, assign, deltas, K, C, nC)


def composer_withheld(world: dict, assign, n: int, rng) -> np.ndarray | None:
    C, nC = world["C"], world["n"]
    i_star = pick_withhold(world, assign)
    v_star = assign[i_star]
    sims = _forced_blocks(world, n, rng, drop=(i_star, v_star))
    deltas, ok = _deltas_from_blocks(sims, nC, C, skip=(i_star, v_star))
    if not ok:
        return None
    # Special recovery of the withheld delta via a within-block relative shift.
    got = None
    others_all = [k for k in range(nC) if k != i_star]
    for j in others_all:
        rest = [k for k in others_all if k != j]
        for u in range(C):
            if u not in sims[j]:
                continue
            dials, R = sims[j][u]
            for w0 in range(C):
                def shift_mode(target):
                    m = (dials[i_star] == target)
                    for k in rest:
                        m &= (dials[k] == w0)
                    return int(np.bincount(R[m], minlength=C).argmax()) if m.sum() >= MIN_CELL else None
                mv, m0 = shift_mode(v_star), shift_mode(REF)
                if mv is not None and m0 is not None:
                    got = (mv - m0) % C
                    break
            if got is not None:
                break
        if got is not None:
            break
    if got is None:
        return None
    deltas[i_star, v_star] = got
    K = _anchor_K(sims, nC, C)  # do(dial_0=REF) survives: we only drop (i*, v*) with v*!=REF
    if K is None:
        return None
    return _predict(world, assign, deltas, K, C, nC)


# ---------------------------------------------------------------------------
# Sweep.
# ---------------------------------------------------------------------------
def _valid_held_queries(world: dict, i_star: int, cap: int, rng) -> list:
    """Exhaustive valid assignments (not all-equal AND true_value != naive sum) restricted to the
    WHOLE-DIAL test cell a[i*] != REF — exactly rung1_train.held_queries. Capped to `cap`, sampled."""
    C, nC = world["C"], world["n"]
    out = []
    def rec(prefix):
        if len(prefix) == nC:
            a = tuple(prefix)
            if len(set(a)) == 1 or a[i_star] == REF:
                return
            if true_value(world, a) == sum(a) % C:
                return
            out.append(a)
            return
        for v in range(C):
            rec(prefix + [v])
    rec([])
    if len(out) > cap:
        idx = rng.choice(len(out), size=cap, replace=False)
        out = [out[k] for k in idx]
    return out


def sweep_cell(n_causes: int, C: int, seed: int, budgets, n_pairs: int) -> dict:
    world = make_world(n_causes, C, seed)
    i_star = holdout_dial(world)
    pairs = _valid_held_queries(world, i_star, n_pairs, np.random.default_rng(seed + 99))
    # Floor band: strongest non-compositional cheater on these queries (context for "comfortably > floor").
    cheats = {"one_cause": [], "modal_obs": [], "nearest_cell": []}
    for a in pairs:
        orc = oracles(world, a)
        for b in cheats:
            cheats[b].append(norm_score(orc[b], orc["oracle"]))
    floor = max(float(np.mean(v)) for v in cheats.values())

    rows = []
    for n in budgets:
        rev, wh, rev_ok, wh_ok = [], [], 0, 0
        for a in pairs:
            orc = oracles(world, a)
            rng_r = np.random.default_rng(seed * 104729 + 3)      # fixed per (seed) -> reproducible
            rng_w = np.random.default_rng(seed * 104729 + 11)
            cr = composer_revealed(world, a, n, rng_r)
            cw = composer_withheld_dial(world, a, n, rng_w)   # §5'.3 WHOLE-DIAL holdout
            rev.append(norm_score(cr, orc["oracle"]) if cr is not None else np.nan)
            wh.append(norm_score(cw, orc["oracle"]) if cw is not None else np.nan)
            rev_ok += cr is not None
            wh_ok += cw is not None
        rows.append({
            "n": n,
            "revealed_mean": float(np.nanmean(rev)) if rev_ok else float("nan"),
            "revealed_ci": _bootstrap_ci(rev),
            "revealed_populated": rev_ok / len(pairs),
            "withheld_mean": float(np.nanmean(wh)) if wh_ok else float("nan"),
            "withheld_ci": _bootstrap_ci(wh),
            "withheld_populated": wh_ok / len(pairs),
        })
    return {"n_causes": n_causes, "C": C, "seed": seed, "n_pairs": len(pairs),
            "holdout_dial": int(i_star),
            "floor": floor, "cheater_means": {b: float(np.mean(v)) for b, v in cheats.items()},
            "budgets": rows}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-causes", type=int, default=3)
    ap.add_argument("--C", type=int, default=5)
    ap.add_argument("--seeds", nargs="+", type=int, default=[20260605, 20260606, 20260607])
    ap.add_argument("--budgets", nargs="+", type=int, default=[100, 200, 400, 800, 1600])
    ap.add_argument("--pairs", type=int, default=30)
    ap.add_argument("--ceiling", type=float, default=0.70, help="pre-registered ceiling bar")
    ap.add_argument("--out-json", type=Path,
                    default=Path(__file__).resolve().parent.parent / "results" / "RUNG1_PREFLIGHT_WHOLEDIAL_2026-06-05.json")
    args = ap.parse_args()

    print("=" * 100)
    print(f"RUNG-1 $0 PRE-FLIGHT (§5'.6 WHOLE-DIAL) — sample-matched composer ceiling "
          f"(revealed vs whole-dial-withheld)")
    print(f"regime: {args.n_causes} dials, C={args.C} | seeds {args.seeds} | N={args.pairs} queries | "
          f"bar>={args.ceiling:.2f}")
    print("=" * 100)

    all_seeds = []
    for s in args.seeds:
        res = sweep_cell(args.n_causes, args.C, s, args.budgets, args.pairs)
        all_seeds.append(res)
        print(f"\nseed {s}  (held dial i*={res['holdout_dial']}, floor band strongest cheater = {res['floor']:+.3f})")
        print(f"  {'rows/blk':>8} | {'REVEALED  [95% CI]':>26} {'pop%':>5} | "
              f"{'WITHHELD  [95% CI]':>26} {'pop%':>5}")
        for r in res["budgets"]:
            rl, rh = r["revealed_ci"]; wl, wh = r["withheld_ci"]
            rstr = f"{r['revealed_mean']:+.3f}[{rl:+.2f},{rh:+.2f}]"
            wstr = f"{r['withheld_mean']:+.3f}[{wl:+.2f},{wh:+.2f}]"
            print(f"  {r['n']:>8} | {rstr:>26} {r['revealed_populated']*100:>4.0f}% | "
                  f"{wstr:>26} {r['withheld_populated']*100:>4.0f}%")

    # Pre-registered min budget: BOTH ceilings >= bar AND both fully populated, on ALL seeds.
    bar = args.ceiling
    def cell_ok(res, n):
        r = next(x for x in res["budgets"] if x["n"] == n)
        return (r["revealed_populated"] == 1.0 and r["withheld_populated"] == 1.0 and
                r["revealed_mean"] >= bar and r["withheld_mean"] >= bar)
    min_budget = next((n for n in args.budgets if all(cell_ok(res, n) for res in all_seeds)), None)

    print("\n" + "=" * 100)
    if min_budget is None:
        print(f">>> NO budget in {args.budgets} keeps BOTH ceilings >= {bar:.2f} fully-populated on all "
              f"{len(args.seeds)} seeds.")
        print(">>> Regime mis-specified at these budgets — widen the sweep or fix the design. DO NOT train.")
    else:
        print(f">>> MIN ROW BUDGET (pre-register this) = {min_budget} forced rows / block: BOTH the "
              f"all-revealed and withheld-effect")
        print(f">>> ceilings are >= {bar:.2f} and fully populated on all {len(args.seeds)} seeds. The "
              f"withheld-effect test is FAIR (ceiling > floor).")
    print("=" * 100)

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    payload = {"regime": {"n_causes": args.n_causes, "C": args.C}, "seeds": args.seeds,
               "budgets": args.budgets, "ceiling_bar": bar, "min_budget": min_budget,
               "per_seed": all_seeds}
    args.out_json.write_text(json.dumps(payload, default=lambda o: o.tolist()
                             if isinstance(o, np.ndarray) else list(o)
                             if isinstance(o, tuple) else o, indent=2) + "\n")
    print(f"Wrote {args.out_json}")


if __name__ == "__main__":
    main()
