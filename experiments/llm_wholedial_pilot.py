"""Whole-dial transport on a real LLM (DeepSeek V4 Flash) — external-validity test of the measurement tool.

Per docs/handoffs/2026-06-06-llm-wholedial-external-validity-handoff.md (red-team REVISE-FIRST + Codex BLOCKER
fixes baked in). Pre-registered regime: 2 dials, C=4, 75 rows/block (whole-dial withheld ceiling +0.935,
results/RUNG1_PREFLIGHT_2DIAL_C4_LLMFEASIBLE_2026-06-06.json).

THE TEST. Show the LLM, in-context: confounded observational rows + the PRESENT dial's clean forced
(interventional) blocks. WITHHOLD the entire clean forced block of dial i*=holdout_dial(world). Ask it to
predict the meter under a NOVEL joint forcing (dial i*=v, dial j=w) with v != REF — so i*'s effect must be
INFERRED cross-block, never read off. Score with the SAME norm_score the tool uses, against the analytic
whole-dial composer (ceiling) and the non-compositional cheaters (floor).

TWO EXCLUSIONS (both required; the 2nd is the Codex leak fix):
  1. whole-dial: drop every clean forced block of i*.
  2. exact-assignment: drop every prompt row (obs or present-forced) whose dial values EXACTLY match any
     query (i*=v, j=w). With 2 dials the present do(j=w) block contains rows where the free dial i* wanders
     to v; R = sum_i table_i[dial_i] + noise, so such a row is a DIRECT sample of the query -> lookup leak.
     Excluding only the exact (v,w) cell preserves identifiability: i*=v still appears in do(j=u != w) blocks,
     which is what cross-block recovery uses.

MANDATORY $0 SELF-TEST GATE (run before ANY API call): the analytic composer, run on EXACTLY the rows that go
into the prompt (same sims object), must score >= 0.80*ceiling (~+0.75), AND no prompt row may match any query
assignment, AND i* must have no forced block. If any fails -> abort, do not spend.

DeepSeek API key: set env DEEPSEEK_API_KEY, or point DEEPSEEK_KEY_PATH at a key file (default ~/.deepseek_key).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import urllib.request
import urllib.error
from pathlib import Path

import numpy as np

# Validated world generator / scorer / oracles (the ramp n-dial version the pre-flight certifies on) ...
from llm_composition_ramp import make_world, simulate, norm_score, oracles, true_value
# ... and the certified whole-dial primitives (we reuse the EXACT recovery math).
from rung1_preflight import (
    REF, MIN_CELL, holdout_dial, _forced_blocks, _deltas_from_blocks, _anchor_K_robust, _predict,
)

KEY_PATH = Path(os.environ.get("DEEPSEEK_KEY_PATH", str(Path.home() / ".deepseek_key")))
API_URL = "https://api.deepseek.com/chat/completions"


# ---------------------------------------------------------------------------
# Query selection — exactly rung1_train.held_queries (a[i*] != REF, not all-equal, true != naive sum).
# ---------------------------------------------------------------------------
def valid_held_queries(world: dict, i_star: int, cap: int, rng) -> list:
    C, nC = world["C"], world["n"]
    out = []
    def rec(prefix):
        if len(prefix) == nC:
            a = tuple(prefix)
            if len(set(a)) == 1 or a[i_star] == REF:
                return
            if true_value(world, a) == sum(a) % C:   # arithmetic-defeating: naive a+b is provably wrong
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


# ---------------------------------------------------------------------------
# Build the prompt's data ONCE (sims + obs), apply BOTH exclusions, and reuse the SAME object for the
# prompt and the self-test composer -> "composer ran on exactly the prompt rows" holds by construction.
# ---------------------------------------------------------------------------
def _filter_rows(block, drop_assignments, nC):
    """block = (dials_list, R). Drop rows whose dial values exactly match ANY assignment in drop_assignments."""
    dials, R = block
    keep = np.ones(len(R), dtype=bool)
    for a in drop_assignments:
        match = np.ones(len(R), dtype=bool)
        for d in range(nC):
            match &= (dials[d] == a[d])
        keep &= ~match
    return ([dials[d][keep] for d in range(nC)], R[keep])


def build_episode(world: dict, queries, n_do: int, n_obs: int, rng, reveal_istar: bool):
    """Returns (sims, obs, i_star). sims[i][v]=(dials,R) forced blocks (i* dropped unless reveal_istar);
    obs=(dials,R) observational. BOTH have the exact-assignment rows for every query removed."""
    nC = world["n"]
    i_star = holdout_dial(world)
    drop_dial = None if reveal_istar else i_star
    sims = _forced_blocks(world, n_do, rng, drop_dial=drop_dial)
    obs = simulate(world, n_obs, rng)  # (dials, R), all dials free (confounded)
    # Exact-assignment exclusion (Codex BLOCKER fix) on every block + obs.
    for i in range(nC):
        for v in list(sims[i].keys()):
            sims[i][v] = _filter_rows(sims[i][v], queries, nC)
    obs = _filter_rows(obs, queries, nC)
    return sims, obs, i_star


# ---------------------------------------------------------------------------
# Row-backed analytic composer over the EXACT episode sims (no fresh simulation).
# Mirrors rung1_preflight.composer_withheld_dial but consumes the passed-in sims.
# ---------------------------------------------------------------------------
def composer_from_sims(world: dict, assign, sims, i_star: int) -> np.ndarray | None:
    C, nC = world["C"], world["n"]
    v_star = assign[i_star]
    if v_star == REF:
        return None
    deltas, ok = _deltas_from_blocks(sims, nC, C, skip_dial=i_star)
    if not ok:
        return None
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


def composer_revealed_from_sims(world: dict, assign, sims) -> np.ndarray | None:
    """All-revealed analytic check (i* block present): every delta read directly; anchor robustly."""
    C, nC = world["C"], world["n"]
    deltas, ok = _deltas_from_blocks(sims, nC, C)
    if not ok:
        return None
    K = _anchor_K_robust(sims, nC, C, 0)  # all dials present; j!=0 anchor is fine, use REF anchor route
    if K is None:
        return None
    return _predict(world, assign, deltas, K, C, nC)


# ---------------------------------------------------------------------------
# Prompt rendering (abstract framing to block pretraining contamination).
# ---------------------------------------------------------------------------
def _fmt_rows(block, nC) -> str:
    dials, R = block
    return "\n".join(" ".join(str(int(dials[d][r])) for d in range(nC)) + f" {int(R[r])}"
                     for r in range(len(R)))


def render_prompt(world: dict, sims, obs, i_star: int, assign, reveal_istar: bool) -> str:
    C, nC = world["C"], world["n"]
    head = " ".join(f"d{d}" for d in range(nC)) + " R"
    forced = []
    for i in range(nC):
        for v in range(C):
            if v in sims[i]:
                forced.append(f"(dial d{i} forced to {v}):\n" + _fmt_rows(sims[i][v], nC))
    qstr = " AND ".join(f"dial d{d} = {assign[d]}" for d in range(nC))
    note = "" if reveal_istar else (
        f"\nNOTE: there is NO block where dial d{i_star} is directly forced — you must infer d{i_star}'s "
        f"effect from how the meter shifts with d{i_star} inside the OTHER forced blocks.")
    return f"""You are analyzing a sealed machine with {nC} dials (d0..d{nC-1}, each an integer 0-{C-1}) and a meter R (integer 0-{C-1}). An internal HIDDEN state you cannot observe nudges the dials. The meter R is set by the dials through a FIXED hidden rule (each dial is first passed through its OWN secret conversion table, then the converted values are combined) plus occasional small noise. "Just adding the dial numbers" will NOT be correct — infer the rule from the data.

OBSERVED data — machine running on its own (dials take whatever the hidden state pushes; CONFOUNDED, so naive dial-vs-meter patterns mislead):
{head}
{_fmt_rows(obs, nC)}

FORCED data — we override one dial to a fixed value (others left free); this breaks the hidden state's pull on the forced dial:
{chr(10).join(forced)}{note}

QUESTION: We now FORCE {qstr} together (a combination NOT shown above). Using the FORCED data to read off each dial's effect (and inferring any dial whose forced block is absent), predict meter R.
Keep reasoning brief and concrete. End with EXACTLY one line: FINAL: <single integer 0-{C-1}>"""


def render_arith_control(world: dict, assign) -> str:
    C, nC = world["C"], world["n"]
    qstr = " AND ".join(f"dial d{d} = {assign[d]}" for d in range(nC))
    return f"""A sealed machine has {nC} dials (d0..d{nC-1}, each 0-{C-1}) and a meter R (0-{C-1}). R is set by a fixed hidden rule of the dials plus small noise. You are given NO data about the rule.
QUESTION: If we force {qstr}, what will meter R read? Give your single best guess.
End with EXACTLY one line: FINAL: <single integer 0-{C-1}>"""


# ---------------------------------------------------------------------------
# DeepSeek API.
# ---------------------------------------------------------------------------
def call_llm(model: str, prompt: str, temperature: float, C: int, max_tokens: int = 16000,
             timeout: int = 300) -> int | None:
    key = os.environ.get("DEEPSEEK_API_KEY") or KEY_PATH.read_text().strip()
    body = json.dumps({"model": model, "messages": [{"role": "user", "content": prompt}],
                       "temperature": temperature, "max_tokens": max_tokens}).encode()
    req = urllib.request.Request(API_URL, data=body,
                                 headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            data = json.loads(r.read())
    except (OSError, TimeoutError, json.JSONDecodeError, KeyError, IndexError) as e:
        # OSError covers urllib.error.URLError AND raw socket errors (e.g. ConnectionResetError) that
        # can fire mid-read after urlopen returns; a transient blip must skip the sample, not kill the run.
        print(f"      [api error] {type(e).__name__}: {e}", flush=True)
        return None
    choice = data["choices"][0]
    content = choice["message"].get("content") or ""
    m = re.findall(r"FINAL:\s*(\d+)", content)
    if not m:
        m = re.findall(r"(\d+)", content)
    if not m:
        print(f"      [no answer] finish={choice.get('finish_reason')} len={len(content)}", flush=True)
        return None
    val = int(m[-1])
    return val if 0 <= val < C else None


def sample_dist(model: str, prompt: str, k: int, temperature: float, C: int,
                max_tokens: int = 16000) -> tuple[np.ndarray, int]:
    counts = np.zeros(C, dtype=np.float64)
    ok = 0
    for i in range(k):
        v = call_llm(model, prompt, temperature, C, max_tokens=max_tokens)
        if v is not None:
            counts[v] += 1; ok += 1
        print(f"      sample {i+1}/{k}: {v}", flush=True)
    return (counts / counts.sum(), ok) if ok else (np.full(C, 1.0 / C), 0)


# ---------------------------------------------------------------------------
# Run.
# ---------------------------------------------------------------------------
def run(args) -> dict:
    world = make_world(args.n_causes, args.C, args.seed)
    C, nC = world["C"], world["n"]
    i_star = holdout_dial(world)
    qrng = np.random.default_rng(args.seed + 99)
    queries = valid_held_queries(world, i_star, args.queries, qrng)
    if args.smoke:
        queries = queries[:1]
    ceiling = args.ceiling
    bar = 0.80 * ceiling
    print(f"world seed={args.seed} regime={nC}d C={C}  held dial i*={i_star}  REF={REF}")
    print(f"queries (a[i*]!=REF, arithmetic-defeating): {queries}")

    # --- Build the WHOLE-DIAL episode once; reuse for prompt + self-test. ---
    erng = np.random.default_rng(args.seed * 104729 + 11)
    sims, obs, i_star = build_episode(world, queries, args.n_do, args.n_obs, erng, reveal_istar=False)

    # ===== MANDATORY $0 SELF-TEST GATE =====
    print("\n=== $0 self-test gate (must pass before any API call) ===")
    # (g1) holdout applied: i* has no forced block.
    assert len(sims[i_star]) == 0, f"LEAK: dial i*={i_star} still has forced blocks {list(sims[i_star])}"
    # (g2) no prompt row matches any query assignment (exact-assignment exclusion worked).
    def any_match(block):
        dials, R = block
        for a in queries:
            m = np.ones(len(R), dtype=bool)
            for d in range(nC):
                m &= (dials[d] == a[d])
            if m.any():
                return True
        return False
    leak = any_match(obs) or any(any_match(sims[i][v]) for i in range(nC) for v in sims[i])
    assert not leak, "LEAK: a prompt row exactly matches a query assignment"
    # (g3) row-backed analytic composer on EXACTLY these rows clears the bar.
    comp_scores = []
    for a in queries:
        pred = composer_from_sims(world, a, sims, i_star)
        orc = oracles(world, a)
        comp_scores.append(norm_score(pred, orc["oracle"]) if pred is not None else np.nan)
    comp_mean = float(np.nanmean(comp_scores))
    pop = float(np.mean([not np.isnan(s) for s in comp_scores]))
    print(f"  holdout ok (i* has 0 forced blocks) ✓   no exact-assignment leak ✓")
    print(f"  row-backed composer on prompt rows: mean={comp_mean:+.3f} populated={pop*100:.0f}%  bar={bar:+.3f}")
    if not (pop == 1.0 and comp_mean >= bar):
        print(f"\n>>> SELF-TEST FAILED: composer {comp_mean:+.3f} (pop {pop*100:.0f}%) < bar {bar:+.3f}. "
              f"Test is unfair at this budget. ABORTING — no API spend.")
        return {"aborted": True, "self_test": {"comp_mean": comp_mean, "pop": pop, "bar": bar}}
    print(f">>> SELF-TEST PASSED. Test is fair on the exact prompt rows; proceeding to API.\n")
    if args.gate_only:
        return {"gate_only": True, "self_test": {"comp_mean": comp_mean, "pop": pop, "bar": bar},
                "n_obs_rows": len(obs[1]), "forced_rows_per_block": args.n_do}

    # --- Build the all-revealed positive-control episode (same format, i* block present). ---
    sims_rev = obs_rev = None
    if args.with_revealed:
        rrng = np.random.default_rng(args.seed * 104729 + 11)  # same seed -> same present-dial rows
        sims_rev, obs_rev, _ = build_episode(world, queries, args.n_do, args.n_obs, rrng, reveal_istar=True)

    rows = []
    for a in queries:
        orc = oracles(world, a)
        print(f"=== query {a}  (true R = {true_value(world, a)}) ===")
        # main whole-dial test
        p_main = render_prompt(world, sims, obs, i_star, a, reveal_istar=False)
        d_main, ok_main = sample_dist(args.model, p_main, args.samples, args.temperature, C, args.max_tokens)
        # no-data arithmetic control (must score low — proves not-just-priors)
        d_ar, ok_ar = sample_dist(args.model, render_arith_control(world, a), args.samples, args.temperature, C, args.max_tokens)
        # all-revealed positive control (OPTIONAL; heavy prompt -> off by default. The whole-dial pass and the
        # prior single-cell +0.936 already evidence format competence.)
        if args.with_revealed:
            p_rev = render_prompt(world, sims_rev, obs_rev, i_star, a, reveal_istar=True)
            d_rev, ok_rev = sample_dist(args.model, p_rev, args.samples, args.temperature, C, args.max_tokens)
        else:
            d_rev, ok_rev = np.full(C, 1.0 / C), -1
        row = {
            "assign": list(a), "true": true_value(world, a),
            "llm_wholedial": norm_score(d_main, orc["oracle"]), "ok_wholedial": ok_main,
            "llm_revealed": norm_score(d_rev, orc["oracle"]), "ok_revealed": ok_rev,
            "llm_arith_nodata": norm_score(d_ar, orc["oracle"]), "ok_arith": ok_ar,
            "composer_ceiling": comp_mean,
            "cheat_one_cause": norm_score(orc["one_cause"], orc["oracle"]),
            "cheat_modal_obs": norm_score(orc["modal_obs"], orc["oracle"]),
            "cheat_nearest_cell": norm_score(orc["nearest_cell"], orc["oracle"]),
            "dist_wholedial": d_main.tolist(), "dist_revealed": d_rev.tolist(), "oracle": orc["oracle"].tolist(),
        }
        rows.append(row)
        print(f"    -> whole-dial={row['llm_wholedial']:+.3f}  revealed(ctrl)={row['llm_revealed']:+.3f}  "
              f"arith-nodata={row['llm_arith_nodata']:+.3f}  ceiling={comp_mean:+.3f}", flush=True)
    return {"model": args.model, "seed": args.seed, "regime": {"n_causes": nC, "C": C},
            "i_star": int(i_star), "n_do": args.n_do, "n_obs": args.n_obs, "bar": bar,
            "ceiling": comp_mean, "rows": rows}


def verdict(res: dict) -> None:
    if res.get("aborted") or res.get("gate_only"):
        return
    rows = res["rows"]
    wd = float(np.nanmean([r["llm_wholedial"] for r in rows]))
    have_rev = all(r["ok_revealed"] >= 0 for r in rows)
    rv = float(np.nanmean([r["llm_revealed"] for r in rows])) if have_rev else float("nan")
    ar = float(np.nanmean([r["llm_arith_nodata"] for r in rows]))
    cheat = float(np.nanmean([max(r["cheat_one_cause"], r["cheat_modal_obs"], r["cheat_nearest_cell"])
                              for r in rows]))
    bar = res["bar"]
    print("\n" + "=" * 80)
    print(f"WHOLE-DIAL LLM TEST — {res['model']}  (ceiling={res['ceiling']:+.3f}, bar={bar:+.3f})")
    print(f"  whole-dial (TEST)     = {wd:+.3f}")
    print(f"  all-revealed (pos ctrl)= {rv:+.3f}" + ("" if have_rev else "  (not run — see prior single-cell +0.936)"))
    print(f"  arith no-data (neg)   = {ar:+.3f}")
    print(f"  strongest cheater     = {cheat:+.3f}")
    if have_rev and rv < bar:
        v = "UNINTERPRETABLE — positive control (all-revealed) also fails => format/tally limit, not transport"
    elif wd >= bar:
        v = "TRANSPORTS — LLM recovers a globally-withheld effect in-context (no side-channel)"
    elif wd <= cheat + 0.15:
        v = "COLLAPSE — LLM composes when shown effects cleanly but does NOT infer the withheld one (memorize-not-infer)"
    else:
        v = "PARTIAL / inconclusive"
    print(f"VERDICT: {v}")
    print("=" * 80)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="deepseek-v4-flash")
    ap.add_argument("--n-causes", type=int, default=2)
    ap.add_argument("--C", type=int, default=4)
    ap.add_argument("--seed", type=int, default=20260606)
    ap.add_argument("--n-do", type=int, default=75)      # forced rows / block (pre-registered FAIR budget)
    ap.add_argument("--n-obs", type=int, default=75)
    ap.add_argument("--queries", type=int, default=3)
    ap.add_argument("--samples", type=int, default=4)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--max-tokens", type=int, default=16000,
                    help="output budget incl. thinking tokens; raise for harder regimes that blow 16k reasoning")
    ap.add_argument("--ceiling", type=float, default=0.935)
    ap.add_argument("--with-revealed", action="store_true",
                    help="also run the heavy all-revealed positive control (off by default)")
    ap.add_argument("--smoke", action="store_true", help="1 query, still full controls")
    ap.add_argument("--gate-only", action="store_true", help="run the $0 self-test gate and stop (no API)")
    ap.add_argument("--out-json", type=Path)
    args = ap.parse_args()

    t0 = time.time()
    res = run(args)
    verdict(res)
    print(f"\nElapsed {(time.time()-t0)/60:.1f} min")
    if args.out_json and not res.get("gate_only"):
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(json.dumps(res, indent=2) + "\n")
        print(f"Wrote {args.out_json}")


if __name__ == "__main__":
    main()
