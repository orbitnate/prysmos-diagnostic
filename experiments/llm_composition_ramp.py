"""LLM composition DIFFICULTY RAMP — Flash vs Pro, with a mandatory $0 per-level gate.

Red-team REVISE-FIRST (2026-06-05): cranking cardinality C / noise / confounding does NOT raise
reasoning difficulty for an additive rule (the answer is always the SAME fixed reconstruction; bigger
C just lengthens the table and fuzzes the oracle — an artifact). The valid lever is INFERENCE-CHAIN
LENGTH: more additive causes => a longer reconstruction (n+1 terms) => more steps a weaker model can
trip on, WITHOUT making the oracle diffuse (oracle stays peaked since all causes are clamped).

Mandatory $0 GATE before any API spend, per level:
  (1) perfect-composer normalized score ~1.0 AND oracle is PEAKED (top-1 >= 0.9) — so a 'cliff' can't
      be oracle-diffuseness/sampling noise.
  (2) the naive shortcut baselines (one-cause, modal-observational, nearest-shown-cell) do NOT score
      high — proving a real reasoning gap exists for a stronger model to win.
Only levels passing BOTH are eligible to run on the LLMs.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
import urllib.request
import urllib.error
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

# DeepSeek API key: set env DEEPSEEK_API_KEY directly, or point DEEPSEEK_KEY_PATH at a file
# (default ~/.deepseek_key). LLM rows require your own key; the $0 gate/validation modes do not.
KEY_PATH = Path(os.environ.get("DEEPSEEK_KEY_PATH", str(Path.home() / ".deepseek_key")))
API_URL = "https://api.deepseek.com/chat/completions"


# ---------------------------------------------------------------------------
# N-cause additive confounded world.
# ---------------------------------------------------------------------------
def make_world(n_causes: int, C: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    tables = [rng.permutation(C) for _ in range(n_causes)]  # secret per-dial conversions
    return {"tables": tables, "n": n_causes, "C": C, "seed": seed}


def simulate(world: dict, n: int, rng, do: dict | None = None):
    """Sample (dials[n_causes][n], R). Hidden H drives every dial (confounding).
    R = (sum_i table_i[dial_i] + noise) mod C."""
    C, tables = world["C"], world["tables"]
    do = do or {}
    H = rng.integers(0, C, size=n)
    dials = []
    for i in range(world["n"]):
        if i in do:
            dials.append(np.full(n, do[i]))
        else:
            noise = np.where(rng.random(n) < 0.85, 0, rng.integers(0, C, n))
            dials.append((H + noise) % C)
    nR = np.where(rng.random(n) < 0.94, 0, rng.integers(0, C, n))
    R = (sum(tables[i][dials[i]] for i in range(world["n"])) + nR) % C
    return dials, R


def dist_of(R: np.ndarray, C: int) -> np.ndarray:
    return np.bincount(R, minlength=C)[:C].astype(np.float64) / len(R)


def norm_score(pred: np.ndarray, oracle: np.ndarray) -> float:
    C = len(oracle); u = 1.0 / C
    U = 1.0 - 0.5 * np.abs(u - oracle).sum()
    mtv = 1.0 - 0.5 * np.abs(pred - oracle).sum()
    denom = 1.0 - U
    return float(np.clip((mtv - U) / denom, -1.0, 1.0)) if denom > 1e-9 else float("nan")


def true_value(world: dict, assign: tuple[int, ...]) -> int:
    return int((sum(world["tables"][i][assign[i]] for i in range(world["n"]))) % world["C"])


def oracles(world: dict, assign: tuple[int, ...], n=40000) -> dict:
    C = world["C"]
    rng = np.random.default_rng(world["seed"] * 7919 + 1)
    obs_dials, obs_R = simulate(world, n, rng)
    joint = dist_of(simulate(world, n, rng, do={i: assign[i] for i in range(world["n"])})[1], C)
    one_cause = dist_of(simulate(world, n, rng, do={0: assign[0]})[1], C)   # ignore other causes
    modal_obs = dist_of(obs_R, C)
    # nearest-shown-cell baseline: most common R among observational rows that match assign on all
    # dials EXCEPT one (the closest cells the data shows for this combo); falls back to modal_obs.
    near = modal_obs
    for drop in range(world["n"]):
        mask = np.ones(n, dtype=bool)
        for i in range(world["n"]):
            if i != drop:
                mask &= (obs_dials[i] == assign[i])
        if mask.sum() >= 5:
            near = dist_of(obs_R[mask], C); break
    return {"oracle": joint, "one_cause": one_cause, "modal_obs": modal_obs, "nearest_cell": near}


# ---------------------------------------------------------------------------
# REAL analytic composer (the genuine positive control).
# Reconstructs each dial's hidden table from FORCED + OBSERVED data ONLY (never the
# joint answer key), composes, and predicts the joint. This is the information CEILING:
# the best a perfect reasoner could do from exactly the data the LLM is shown.
#   - relative shift  g_i[v]-g_i[r]  from the cyclic shift between do(dial_i=v) and do(dial_i=r)
#     (the free-dial background is identical under both, so it cancels in distribution);
#   - absolute anchor  sum_i g_i[r]  from observational rows where ALL dials equal r
#     (R there = sum_i g_i[r] + small noise);
#   - predict point mass at (anchor + sum_i (g_i[a_i]-g_i[r])) mod C.
# If this clears >=0.90 the positive control is genuine; if it falls short, that shortfall
# IS the forced-data information ceiling the LLM must be judged against.
# ---------------------------------------------------------------------------
def analytic_composer(world: dict, assign: tuple[int, ...], n: int = 40000,
                      use_forced: bool = True) -> np.ndarray | None:
    """Reconstruct using the SAME per-row (dials, R) data the LLM is shown. Under do(dial_i=v),
    CONDITION on the other dials all equal to a reference w0: then R = g_i[v] + (const) + noise,
    so its mode isolates g_i[v] up to a per-dial constant (the shared hidden state cancels exactly
    because both v and the reference are conditioned on the identical background).

    ABLATION (use_forced=False): deny the interventional/forced data. Without do() data the per-dial
    effects are unrecoverable under confounding, so deltas stay 0 and it can only anchor an absolute
    level from observational all-equal rows -> it predicts the SAME value for every assignment and
    cannot compose. This is the composition-specific negative control (Control #4)."""
    C, nC = world["C"], world["n"]
    rng = np.random.default_rng(world["seed"] * 104729 + 3)
    r = 0  # per-dial reference value
    sims = [{v: simulate(world, n, rng, do={i: v}) for v in range(C)} for i in range(nC)]

    def cond_mode(i: int, v: int, w0: int):
        dials, R = sims[i][v]
        m = np.ones(n, dtype=bool)
        for j in range(nC):
            if j != i:
                m &= (dials[j] == w0)
        return int(np.bincount(R[m], minlength=C).argmax()) if m.sum() >= 10 else None

    deltas = np.zeros((nC, C), dtype=int)  # deltas[i, v] = g_i[v] - g_i[r]  (mod C)
    if use_forced:
        for i in range(nC):
            w0 = next((c for c in range(C) if cond_mode(i, r, c) is not None), None)
            if w0 is None:
                return None
            base = cond_mode(i, r, w0)
            for v in range(C):
                mv = cond_mode(i, v, w0)
                if mv is None:
                    return None
                deltas[i, v] = (mv - base) % C
    # absolute anchor K = sum_i g_i[r]: from do(dial_0=r) rows with ALL other dials == r,
    # R = sum_i g_i[r] + noise. Fall back to observational all-equal rows if too few
    # (the ONLY route when forced data is ablated).
    K = cond_mode(0, r, r) if use_forced else None
    if K is None:
        obs_dials, obs_R = simulate(world, n, rng)
        for h in range(C):
            m = np.ones(n, dtype=bool)
            for i in range(nC):
                m &= (obs_dials[i] == h)
            if m.sum() >= 20:
                Fh = int(np.bincount(obs_R[m], minlength=C).argmax())
                K = (Fh - sum(int(deltas[i, h]) for i in range(nC))) % C
                break
    if K is None:
        return None  # data cannot anchor the absolute level -> a genuine information-ceiling gap
    pred_val = (K + sum(int(deltas[i, assign[i]]) for i in range(nC))) % C
    pred = np.zeros(C)
    pred[pred_val] = 1.0
    return pred


def _bootstrap_ci(vals, reps: int = 10000, seed: int = 0) -> tuple[float, float]:
    vals = np.asarray([v for v in vals if not np.isnan(v)], dtype=float)
    if len(vals) == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    means = vals[rng.integers(0, len(vals), size=(reps, len(vals)))].mean(axis=1)
    return (float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5)))


# ---------------------------------------------------------------------------
# $0 per-level gate.
# ---------------------------------------------------------------------------
def select_pairs(world: dict, n_pairs: int, rng) -> list[tuple[int, ...]]:
    """Test assignments where the true answer != the naive sum-of-raw-dials (defeats arithmetic),
    and that are not all-equal (so confounding-correlated data doesn't trivially reveal them)."""
    C, nC = world["C"], world["n"]
    cands = []
    for _ in range(2000):
        a = tuple(int(x) for x in rng.integers(0, C, nC))
        if len(set(a)) == 1:
            continue
        if true_value(world, a) == sum(a) % C:
            continue
        cands.append(a)
    seen, uniq = set(), []
    for a in cands:
        if a not in seen:
            seen.add(a); uniq.append(a)
        if len(uniq) >= n_pairs:
            break
    return uniq


CHEATERS = ("one_cause", "modal_obs", "nearest_cell")


def gate_level(n_causes: int, C: int, seed: int, n_pairs: int = 30) -> dict:
    world = make_world(n_causes, C, seed)
    rng = np.random.default_rng(seed + 99)
    pairs = select_pairs(world, n_pairs, rng)
    rows = []
    for a in pairs:
        orc = oracles(world, a)
        comp = analytic_composer(world, a)
        rows.append({
            "assign": a,
            "oracle_top1": float(orc["oracle"].max()),
            # GENUINE positive control: reconstructed-from-data composer vs the joint oracle.
            "composer": norm_score(comp, orc["oracle"]) if comp is not None else float("nan"),
            "one_cause": norm_score(orc["one_cause"], orc["oracle"]),
            "modal_obs": norm_score(orc["modal_obs"], orc["oracle"]),
            "nearest_cell": norm_score(orc["nearest_cell"], orc["oracle"]),
        })
    top1 = float(np.mean([r["oracle_top1"] for r in rows]))
    composer_mean = float(np.nanmean([r["composer"] for r in rows]))
    composer_ci = _bootstrap_ci([r["composer"] for r in rows])
    # Per-baseline MEAN + bootstrap CI, each thresholded separately (NO max-over-baselines).
    baselines = {b: float(np.mean([r[b] for r in rows])) for b in CHEATERS}
    baseline_ci = {b: _bootstrap_ci([r[b] for r in rows]) for b in CHEATERS}
    worst_baseline = max(baselines.values())
    peaked = top1 >= 0.90
    composer_ok = composer_mean >= 0.90
    cheaters_ok = worst_baseline <= 0.10
    return {"n_causes": n_causes, "C": C, "n_pairs": len(rows), "oracle_top1": top1,
            "composer_mean": composer_mean, "composer_ci": composer_ci,
            "baselines": baselines, "baseline_ci": baseline_ci, "worst_baseline": worst_baseline,
            "peaked": peaked, "composer_ok": composer_ok, "cheaters_ok": cheaters_ok,
            "passes": bool(peaked and composer_ok and cheaters_ok), "pairs": pairs, "rows": rows}


def run_gate(levels: list[tuple[int, int]], seed: int, n_pairs: int = 30) -> list[dict]:
    print("\n" + "=" * 92)
    print(f"$0 PER-LEVEL GATE (no API) — seed {seed}, N={n_pairs} puzzles/level")
    print("=" * 92)
    print(f"{'level':>14} | {'oracle_top1':>11} {'composer[95% CI]':>22} | "
          f"{'one_cause':>10} {'modal_obs':>10} {'nearest':>10} | {'PASS':>5}")
    print("-" * 92)
    out = []
    for (nc, C) in levels:
        g = gate_level(nc, C, seed, n_pairs)
        out.append(g)
        cl, ch = g["composer_ci"]
        b = g["baselines"]
        comp_str = f"{g['composer_mean']:+.3f}[{cl:+.2f},{ch:+.2f}]"
        print(f"{f'{nc}c C={C}':>14} | {g['oracle_top1']:>11.3f} {comp_str:>22} | "
              f"{b['one_cause']:>+10.3f} {b['modal_obs']:>+10.3f} {b['nearest_cell']:>+10.3f} | "
              f"{str(g['passes']):>5}")
    print("-" * 92)
    print("PASS = oracle peaked (top1>=0.90) AND real reconstructed composer >=0.90 AND EVERY cheater")
    print("baseline mean <=0.10 (each thresholded separately — no max-over-baselines inflation).")
    print("Only PASS levels are eligible to spend API on.")
    print("=" * 92)
    return out


# ---------------------------------------------------------------------------
# Analytic validity controls #3 (graded), #4 (ablation), #5 (shortcut-bait). All $0.
# ---------------------------------------------------------------------------
MIX_FRACS = (0.0, 0.25, 0.5, 0.75, 1.0)


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    rx = np.argsort(np.argsort(x)).astype(float)
    ry = np.argsort(np.argsort(y)).astype(float)
    rx -= rx.mean(); ry -= ry.mean()
    denom = np.sqrt((rx ** 2).sum() * (ry ** 2).sum())
    return float((rx * ry).sum() / denom) if denom > 1e-9 else float("nan")


def validate_controls(n_causes: int, C: int, seed: int, n_pairs: int = 30) -> dict:
    """Controls #3/#4/#5, pre-registered in docs/TOOL_VALIDATION_PROTOCOL_2026-06-05.md."""
    world = make_world(n_causes, C, seed)
    rng = np.random.default_rng(seed + 99)
    pairs = select_pairs(world, n_pairs, rng)
    u = np.full(C, 1.0 / C)
    full, ablated = [], []
    mix = {f: [] for f in MIX_FRACS}
    mem_on, mem_off = [], []  # composer is invariant to the cell -> comp_on == comp_off, Δ=0
    for a in pairs:
        orc = oracles(world, a)
        oracle = orc["oracle"]
        comp = analytic_composer(world, a, use_forced=True)
        if comp is None:
            continue
        abl = analytic_composer(world, a, use_forced=False)
        cs = norm_score(comp, oracle)
        full.append(cs)
        ablated.append(norm_score(abl, oracle) if abl is not None else norm_score(u, oracle))
        for f in MIX_FRACS:                                   # #3: blend composer with the floor
            mix[f].append(norm_score(f * comp + (1 - f) * u, oracle))
        mem_on.append(norm_score(oracle, oracle))             # #5: cell shown -> exact memorize = 1
        mem_off.append(norm_score(orc["nearest_cell"], oracle))  # cell withheld -> memorize fails ~0
    full, ablated = np.array(full), np.array(ablated)
    mix_means = {f: float(np.mean(mix[f])) for f in MIX_FRACS}
    fr = np.array(MIX_FRACS); mm = np.array([mix_means[f] for f in MIX_FRACS])
    rho = _spearman(fr, mm)
    monotone = bool(np.all(np.diff(mm) >= -1e-9))
    drop = float(full.mean() - ablated.mean())
    mem_jump = float(np.mean(mem_on) - np.mean(mem_off))
    comp_delta = 0.0  # composer never reads the test cell, by construction
    return {
        "n_causes": n_causes, "C": C, "n_pairs": int(len(full)),
        "c3_mixture_means": mix_means, "c3_spearman": rho, "c3_monotone": monotone,
        "c3_pass": bool(rho >= 0.95 and monotone),
        "c4_full": float(full.mean()), "c4_ablated": float(ablated.mean()), "c4_drop": drop,
        "c4_pass": bool(drop >= 0.50),
        "c5_mem_on": float(np.mean(mem_on)), "c5_mem_off": float(np.mean(mem_off)),
        "c5_mem_jump": mem_jump, "c5_composer_delta": comp_delta,
        "c5_pass": bool(mem_jump >= 0.80 and comp_delta <= 0.10),
    }


def run_validation(levels: list[tuple[int, int]], seeds: list[int], n_pairs: int = 30) -> list[dict]:
    print("\n" + "=" * 92)
    print("$0 VALIDITY CONTROLS #3 (graded) / #4 (ablation) / #5 (shortcut-bait)")
    print("=" * 92)
    print(f"{'seed/level':>16} | {'#3 mix ρ':>9} {'mono':>5} | {'#4 full→abl (drop)':>22} | "
          f"{'#5 memON→OFF':>14} | {'3':>1} {'4':>1} {'5':>1}")
    print("-" * 92)
    out = []
    for s in seeds:
        for (nc, C) in levels:
            v = validate_controls(nc, C, s, n_pairs)
            v["seed"] = s
            out.append(v)
            c4 = f"{v['c4_full']:+.3f}->{v['c4_ablated']:+.3f}({v['c4_drop']:+.2f})"
            c5 = f"{v['c5_mem_off']:+.3f}->{v['c5_mem_on']:+.2f}"
            flags = ("Y" if v["c3_pass"] else "n", "Y" if v["c4_pass"] else "n",
                     "Y" if v["c5_pass"] else "n")
            print(f"{f'{s} {nc}c':>16} | {v['c3_spearman']:>9.3f} {str(v['c3_monotone']):>5} | "
                  f"{c4:>22} | {c5:>14} | {flags[0]:>1} {flags[1]:>1} {flags[2]:>1}")
    print("-" * 92)
    print("#3 PASS: Spearman(f,score)>=0.95 & monotone. #4 PASS: composer drops >=0.50 when forced")
    print("data ablated. #5 PASS: memorizer jumps >=0.80 when answer shown; composer flat (Δ=0).")
    print("=" * 92)
    return out


# ---------------------------------------------------------------------------
# Prompt + API (for the actual LLM run on gated levels).
# ---------------------------------------------------------------------------
def render(world: dict, assign: tuple[int, ...], rng, n_obs=14, n_do=3,
          with_forced: bool = True) -> str:
    """with_forced=False is the ABLATED prompt: identical observational data, but the FORCED
    (interventional) blocks are removed. A genuine composer needs the forced data to isolate each
    dial's effect under confounding, so its score should DROP; a model that only pattern-matches the
    observed rows is unaffected. This is the within-model ground-truth-direction test (protocol §5)."""
    C, nC = world["C"], world["n"]
    names = [f"D{i+1}" for i in range(nC)]
    def rows(dials, R, k, exclude_assign):
        keep = np.ones(len(R), dtype=bool)
        if exclude_assign is not None:
            m = np.ones(len(R), dtype=bool)
            for i in range(nC):
                m &= (dials[i] == exclude_assign[i])
            keep &= ~m
        idx = np.where(keep)[0]
        idx = rng.choice(idx, size=min(k, len(idx)), replace=False)
        return "\n".join(" ".join(str(int(dials[i][j])) for i in range(nC)) + f"  {int(R[j])}"
                         for j in idx)
    obs = simulate(world, 600, rng)
    header = " ".join(names) + "  R"
    q = ", ".join(f"{names[i]}={assign[i]}" for i in range(nC))
    intro = (f"""You are analyzing a sealed machine with {nC} dials ({', '.join(names)}, each integer """
             f"""0-{C-1}) and a meter R (0-{C-1}). A HIDDEN internal state nudges the dials. R is set """
             f"""by a FIXED hidden rule: each dial passes through its OWN secret conversion, the """
             f"""conversions are summed, then small noise. "Just adding the dial numbers" is WRONG.""")
    obs_block = (f"\n\nOBSERVED data (machine running freely; CONFOUNDED by the hidden state — naive "
                 f"patterns mislead):\n{header}\n{rows(*obs, n_obs, assign)}")
    if with_forced:
        forced_blocks = []
        for i in range(nC):
            for v in range(C):
                d = simulate(world, 200, rng, do={i: v})
                forced_blocks.append(f"(dial {names[i]} forced to {v}):\n" + rows(*d, n_do, assign))
        forced_block = ("\n\nFORCED data (we override ONE dial; the others run free — this reveals "
                        f"that dial's effect):\n{chr(10).join(forced_blocks)}")
        instruct = ("Read each dial's effect from the FORCED data, combine them, and predict R.")
    else:
        forced_block = ""
        instruct = ("Infer each dial's effect from the OBSERVED data alone, combine them, and predict R.")
    return (f"{intro}{obs_block}{forced_block}\n\nQUESTION: We now FORCE all dials together: {q} "
            f"(this exact combination is NOT shown above). {instruct} Keep reasoning brief and "
            f"concrete.\nEnd with EXACTLY one line: FINAL: <single integer 0-{C-1}>")


def call_llm(model: str, prompt: str, temperature: float, C: int,
             max_tokens=64000, timeout=900, retries=4) -> int | None:
    key = os.environ.get("DEEPSEEK_API_KEY") or KEY_PATH.read_text().strip()
    body = json.dumps({"model": model, "messages": [{"role": "user", "content": prompt}],
                       "temperature": temperature, "max_tokens": max_tokens}).encode()
    data = None
    for attempt in range(retries):
        req = urllib.request.Request(API_URL, data=body,
                                     headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
        try:  # catch ALL transient errors (URLError, TimeoutError, IncompleteRead, ConnectionReset, ...)
            with urllib.request.urlopen(req, timeout=timeout) as r:
                data = json.loads(r.read())
            break
        except Exception as e:  # noqa: BLE001 - a single flaky call must not kill the whole run
            print(f"      [api retry {attempt+1}/{retries}] {type(e).__name__}: {e}", flush=True)
            if attempt == retries - 1:
                return None
            time.sleep(2 * (attempt + 1))
    if data is None:
        return None
    choice = data["choices"][0]; content = choice["message"].get("content") or ""
    m = re.findall(r"FINAL:\s*(\d+)", content) or re.findall(r"(\d+)", content)
    if not m:
        print(f"      [no answer] finish={choice.get('finish_reason')}", flush=True); return None
    v = int(m[-1]); return v if 0 <= v < C else None


def run_level(model: str, world: dict, pairs, k: int, temperature: float,
              with_forced: bool = True, max_workers: int = 16) -> dict:
    """All (pair x sample) calls are independent & stateless -> run them concurrently.
    Prompts are pre-rendered serially (rng is not thread-safe); each sample gets fresh data rows.
    with_forced=False renders the ABLATED prompt (no interventional data)."""
    cond = "forced" if with_forced else "ABLATED"
    rng = np.random.default_rng(world["seed"] + 7)
    tasks = [(a, render(world, a, rng, with_forced=with_forced)) for a in pairs for _ in range(k)]
    answers: dict = {a: [] for a in pairs}
    done = [0]
    total = len(tasks)

    def work(item):
        a, prompt = item
        v = call_llm(model, prompt, temperature, world["C"])
        done[0] += 1
        print(f"      {model}[{cond}] n={world['n']} call {done[0]}/{total} pair={a}: {v}", flush=True)
        return a, v

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        for a, v in ex.map(work, tasks):
            answers[a].append(v)

    rows = []
    for a in pairs:
        orc = oracles(world, a)
        counts = np.zeros(world["C"])
        for v in answers[a]:
            if v is not None:
                counts[v] += 1
        llm = counts / counts.sum() if counts.sum() else np.full(world["C"], 1.0 / world["C"])
        rows.append({"assign": a, "llm": norm_score(llm, orc["oracle"]),
                     "n_valid": int(sum(v is not None for v in answers[a]))})
    per_puzzle = [r["llm"] for r in rows]
    lo, hi = _bootstrap_ci(per_puzzle)
    return {"model": model, "condition": cond, "n_causes": world["n"], "C": world["C"],
            "mean_llm": float(np.mean(per_puzzle)), "ci": (lo, hi),
            "mean_valid_frac": float(np.mean([r["n_valid"] / k for r in rows])),
            "per_puzzle": per_puzzle, "rows": rows}


def _paired_drop_ci(a, b, reps: int = 10000, seed: int = 1):
    d = np.asarray(a, float) - np.asarray(b, float)
    rng = np.random.default_rng(seed)
    means = d[rng.integers(0, len(d), size=(reps, len(d)))].mean(axis=1)
    return float(d.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--validate", action="store_true",
                    help="$0: full validity controls #1-#5 across master seeds, no API")
    ap.add_argument("--ablation", action="store_true",
                    help="real-LLM within-model ablation (with vs without forced data); spends API")
    ap.add_argument("--gate-only", action="store_true", help="$0: just the per-level gate, no API")
    ap.add_argument("--models", nargs="+", default=["deepseek-v4-flash", "deepseek-v4-pro"])
    ap.add_argument("--levels", nargs="+", default=["2,5", "3,5", "4,5"],
                    help="comma pairs n_causes,C (e.g. 2,5 3,5 4,5)")
    ap.add_argument("--samples", type=int, default=6)
    ap.add_argument("--pairs", type=int, default=30)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--seed", type=int, default=20260605)
    ap.add_argument("--seeds", nargs="+", type=int, default=[20260605, 20260606, 20260607])
    ap.add_argument("--abl-level", default="3,5", help="n_causes,C for the LLM ablation demo")
    ap.add_argument("--out-json", type=Path)
    args = ap.parse_args()

    levels = [(int(x.split(",")[0]), int(x.split(",")[1])) for x in args.levels]
    payload: dict = {}

    # ---- $0 controls #1 & #2 (the per-level gate), reported per master seed ----
    all_gates = {}
    for s in args.seeds:
        all_gates[s] = run_gate(levels, s, args.pairs)
    payload["gate_by_seed"] = all_gates

    # ---- $0 controls #3/#4/#5 ----
    if args.validate:
        payload["controls_345"] = run_validation(levels, args.seeds, args.pairs)
        gate_ok = all(g["passes"] for gl in all_gates.values() for g in gl)
        c_ok = all(v["c3_pass"] and v["c4_pass"] and v["c5_pass"] for v in payload["controls_345"])
        verdict = "VALIDATED" if (gate_ok and c_ok) else "NOT VALIDATED"
        print(f"\n>>> $0 INSTRUMENT-VALIDITY VERDICT (all 5 controls, all {len(args.seeds)} seeds): "
              f"{verdict} <<<")
        payload["validity_verdict"] = verdict

    # ---- real-LLM within-model ablation (protocol §5) ----
    if args.ablation:
        nc, C = (int(x) for x in args.abl_level.split(","))
        world = make_world(nc, C, args.seed)
        pairs = gate_level(nc, C, args.seed, args.pairs)["pairs"]
        print(f"\n{'='*70}\nREAL-LLM ABLATION — {nc} causes C={C}, {len(pairs)} puzzles x "
              f"{args.samples} samples, models {args.models}\n{'='*70}")
        abl = []
        for model in args.models:
            cells = {}
            for wf in (True, False):
                r = run_level(model, world, pairs, args.samples, args.temperature, with_forced=wf)
                cells[wf] = r; abl.append(r)
                print(f"  -> {model} [{'WITH' if wf else 'WITHOUT'} forced]: "
                      f"{r['mean_llm']:+.3f} CI[{r['ci'][0]:+.2f},{r['ci'][1]:+.2f}] "
                      f"(answered {r['mean_valid_frac']*100:.0f}%)", flush=True)
            d, dlo, dhi = _paired_drop_ci(cells[True]["per_puzzle"], cells[False]["per_puzzle"])
            composes = (cells[True]["mean_llm"] >= 0.30) and (dlo > 0.0)
            print(f"  === {model}: DROP when forced data removed = {d:+.3f} CI[{dlo:+.2f},{dhi:+.2f}]"
                  f"  -> genuine composition: {'YES' if composes else 'no'}", flush=True)
            cells[True]["drop"] = {"mean": d, "ci": (dlo, dhi), "composes": bool(composes)}
        payload["ablation"] = abl

    if args.out_json:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        def _enc(o):
            if isinstance(o, np.ndarray):
                return o.tolist()
            if isinstance(o, (tuple, np.generic)):
                return o.tolist() if isinstance(o, np.generic) else list(o)
            return o
        args.out_json.write_text(json.dumps(payload, default=_enc, indent=2) + "\n")
        print(f"\nWrote {args.out_json}")


if __name__ == "__main__":
    main()
