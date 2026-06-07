"""System Build — Rung 1.5 (architecture-injected): conv-combine + learned reads + WHOLE-DIAL TRANSPORT.

Pre-registered in docs/SYSTEM_RUNG1_SPEC.md §5'/§B'/§D' (Codex-reviewed; red-team GO). Runs LOCAL on the
DGX Spark, $0, no API. Tests the ONE open axis: does a trained model's deconfounded read TRANSPORT to a
per-dial effect whose ENTIRE clean forced block was GLOBALLY withheld (must infer cross-block, not look up
nor eliminate)?

Rung 1.5 design (spec §5'):
  - World: the VALIDATED llm_composition_ramp additive-mod-C machine (reuse make_world/simulate/oracles).
  - Architecture (§5'.1): per-dial query tokens cross-attend to episode rows, each emitting a distribution
    p_i over C (its estimate of g_i(a_i)); COMBINE = EXACT circular convolution of the p_i (the mod-C add
    operator we OWN — fixed, differentiable, no grokking). Order-invariant ⇒ the verdict certifies the
    deconfounded READ (incl. the inferred withheld factor), NOT learned composition. <<1M params.
  - WHOLE-DIAL global holdout H[world]=i* (§5'.3, Codex [CDX-pc]): EVERY clean forced block of dial i* is
    withheld from all training for that world (elimination impossible — one-value withholding was trivially
    solvable on a bijection). Training queries hold a[i*]=REF (no label depends on g_{i*}[non-REF]); the
    PRIMARY test queries set a[i*]!=REF ⇒ must infer cross-block (transport).
  - Two arms (§5'.2): aux-OFF = HEADLINE (only the joint is supervised; reads LEARNED). aux-ON = CEILING /
    fairness control only (per-dial reads supervised on NON-held dials; a PASS here is NOT transport
    evidence). aux is query-token-keyed and asserted to NEVER touch the held dial i* (§5'.4).
  - Whole-dial skill withholding (§5'.3.4): with prob 0.5 a training episode ALSO withholds an entire
    random non-i* dial while the query uses it at non-REF — teaches in-context inference of a fully-absent
    dial, the exact procedure applied to i* at test.
  - Exact targets [CDX-7]: true class -> 0.94 + 0.06/C, others -> 0.06/C (the 94%/6% R-noise).

This file: data-gen (+ assertions), conv+aux model, training, and the eval grid w/ controls. SMOKE config
by default (fast pipeline check); --full runs the pre-registered budget (800 forced rows/block @3 dials).
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from llm_composition_ramp import make_world, simulate, norm_score, oracles, true_value

REF = 0
# Sample-matched analytic ceiling, certified by rung1_preflight.py §5'.6 (whole-dial, 800 rows/blk, 3 seeds).
CEILING = 0.937


# ---------------------------------------------------------------------------
# Worlds, valid queries, and the GLOBAL WHOLE-DIAL held-out registry H (spec §5'.3).
# ---------------------------------------------------------------------------
def valid_queries(world: dict) -> list[tuple[int, ...]]:
    """Exhaustive valid assignments: not all-equal AND true_value != naive sum (defeats arithmetic)."""
    C, nC = world["C"], world["n"]
    out = []
    def rec(prefix):
        if len(prefix) == nC:
            a = tuple(prefix)
            if len(set(a)) == 1:
                return
            if true_value(world, a) == sum(a) % C:
                return
            out.append(a)
            return
        for v in range(C):
            rec(prefix + [v])
    rec([])
    return out


def holdout_for(world: dict) -> int:
    """WHOLE-DIAL holdout i* (§5'.3, supersedes the §4 one-value cell). Deterministic per world: a fixed
    hash of the world seed (chosen blind to outcomes). EVERY clean forced block of dial i* is withheld for
    that world, so g_{i*}[.] can only be reached by cross-block inference, never by permutation-elimination."""
    h = (world["seed"] * 2654435761) & 0xFFFFFFFF
    return int(h % world["n"])


# ---------------------------------------------------------------------------
# Per-world row POOLS (simulate once, sample per episode -> cheap + reproducible).
# ---------------------------------------------------------------------------
class WorldPool:
    def __init__(self, world: dict, pool: int, seed_off: int):
        C, nC = world["C"], world["n"]
        rng = np.random.default_rng(world["seed"] * 100003 + seed_off)
        od, oR = simulate(world, pool, rng)
        self.obs = (np.stack(od, 1).astype(np.int64), oR.astype(np.int64))  # (dials[pool,nC], R[pool])
        self.forced = {}
        for i in range(nC):
            for v in range(C):
                d, R = simulate(world, pool, rng, do={i: v})
                self.forced[(i, v)] = (np.stack(d, 1).astype(np.int64), R.astype(np.int64))
        self.world = world

    def sample_obs(self, k, rng):
        d, R = self.obs
        idx = rng.integers(0, len(R), size=k)
        return d[idx], R[idx]

    def sample_forced(self, i, v, k, rng):
        d, R = self.forced[(i, v)]
        idx = rng.integers(0, len(R), size=k)
        return d[idx], R[idx]


# ---------------------------------------------------------------------------
# Episode tensor: integer features per row [type, forced_dial, forced_val, dial_0..n-1, R].
# NEVER renders text. `omit` = set of (dial,value) forced cells to leave OUT (always includes the WHOLE
# global-H dial; may also include a whole skill-withheld dial).
# ---------------------------------------------------------------------------
FEAT_TYPE, FEAT_FDIAL, FEAT_FVAL = 0, 1, 2  # then nC dial columns, then R


def build_episode(pool: WorldPool, assign, n_obs: int, n_do: int, omit, rng):
    world = pool.world
    C, nC = world["C"], world["n"]
    omit = set(omit)
    rows = []
    od, oR = pool.sample_obs(n_obs, rng)
    for j in range(n_obs):
        rows.append([0, nC, C] + list(od[j]) + [oR[j]])  # type=0 obs; fdial=nC (none); fval=C (none)
    for i in range(nC):
        for v in range(C):
            if (i, v) in omit:
                continue
            fd, fR = pool.sample_forced(i, v, n_do, rng)
            for j in range(n_do):
                rows.append([1, i, v] + list(fd[j]) + [fR[j]])
    arr = np.asarray(rows, dtype=np.int64)
    q = np.asarray(assign, dtype=np.int64)
    return arr, q


def pad_batch(rows_list):
    """Pad a list of [n_i, F] row arrays to a common N. Returns (R [B,N,F] int64, pad_mask [B,N] bool,
    True where padded). The mask is REQUIRED: zero-padding rows otherwise look like real type=0 obs rows
    and would pollute the attention (skill-withholding makes episodes vary in length)."""
    N = max(r.shape[0] for r in rows_list)
    B, F = len(rows_list), rows_list[0].shape[1]
    R = np.zeros((B, N, F), dtype=np.int64)
    mask = np.ones((B, N), dtype=bool)
    for k, r in enumerate(rows_list):
        R[k, :r.shape[0]] = r
        mask[k, :r.shape[0]] = False
    return R, mask


def exact_target(world: dict, assign) -> np.ndarray:
    C = world["C"]
    t = np.full(C, 0.06 / C)
    t[true_value(world, assign)] = 0.94 + 0.06 / C
    return t


def aux_targets(world: dict, assign) -> np.ndarray:
    """Per-dial read target g_p[a_p] = tables[p][a_p] (the contribution of dial p under the query)."""
    return np.asarray([world["tables"][p][assign[p]] for p in range(world["n"])], dtype=np.int64)


# ---------------------------------------------------------------------------
# Rung 1.5 model (<<1M params): per-dial read heads + EXACT circular-convolution combine (§5'.1).
# ---------------------------------------------------------------------------
class Rung1Model(nn.Module):
    def __init__(self, nC: int, C: int, d: int = 64, heads: int = 4, layers: int = 2):
        super().__init__()
        self.nC, self.C, self.d = nC, C, d
        self.type_emb = nn.Embedding(2, d)
        self.fdial_emb = nn.Embedding(nC + 1, d)       # 0..nC-1 forced dial, nC = none
        self.fval_emb = nn.Embedding(C + 1, d)         # 0..C-1 forced val, C = none
        self.dial_emb = nn.ModuleList([nn.Embedding(C, d) for _ in range(nC)])  # per-position dial value
        self.R_emb = nn.Embedding(C, d)
        # SHARED query-value embedding across dials: training holds the held dial at REF, so a per-dial
        # q_emb[i*] would leave non-REF rows UNTRAINED -> the withheld test would hit a cold-start random
        # embedding (an artifact, not transport). Sharing means q_emb[v!=REF] is trained via the other
        # dials and reused for i*; dial identity is still carried by the per-dial q_base token below.
        self.q_emb = nn.Embedding(C, d)
        self.q_base = nn.Parameter(torch.randn(1, nC, d) * 0.02)                # one query token PER DIAL
        self.layers = nn.ModuleList([nn.ModuleDict({
            "attn": nn.MultiheadAttention(d, heads, batch_first=True),
            "ln1": nn.LayerNorm(d), "ln2": nn.LayerNorm(d),
            "ff": nn.Sequential(nn.Linear(d, 2 * d), nn.GELU(), nn.Linear(2 * d, d)),
        }) for _ in range(layers)])
        # Per-dial READ head (shared): query token p -> logits over C = estimate of g_p(a_p). The COMBINE
        # is the fixed circular convolution below (NO learned params) -> the net only learns the read.
        self.read_head = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d), nn.GELU(), nn.Linear(d, C))

    def embed_rows(self, rows):  # rows [B,N,F]
        nC = self.nC
        e = self.type_emb(rows[..., FEAT_TYPE]) + self.fdial_emb(rows[..., FEAT_FDIAL]) \
            + self.fval_emb(rows[..., FEAT_FVAL]) + self.R_emb(rows[..., 3 + nC])
        for p in range(nC):
            e = e + self.dial_emb[p](rows[..., 3 + p])
        return e  # [B,N,d]

    def _circ_conv(self, ps):  # ps [B,nC,C] probabilities -> joint [B,C]
        C = self.C
        joint = ps[:, 0, :]
        for p in range(1, self.nC):
            pj = ps[:, p, :]
            new = torch.zeros_like(joint)
            for a in range(C):
                new = new + joint[:, a:a + 1] * torch.roll(pj, shifts=a, dims=1)  # new[k]+=joint[a]*pj[k-a]
            joint = new
        return joint

    def forward(self, rows, query, pad_mask=None):  # rows [B,N,F], query [B,nC]; pad_mask [B,N] True=pad
        B = rows.shape[0]
        mem = self.embed_rows(rows)                          # [B,N,d]
        q = self.q_base.expand(B, self.nC, self.d).clone()   # [B,nC,d] one token per dial
        for p in range(self.nC):
            q[:, p] = q[:, p] + self.q_emb(query[:, p])       # shared query-value embedding
        for L in self.layers:
            a, _ = L["attn"](L["ln1"](q), mem, mem, key_padding_mask=pad_mask, need_weights=False)
            q = q + a
            q = q + L["ff"](L["ln2"](q))
        read_logits = self.read_head(q)                      # [B,nC,C] per-dial read
        read_p = F.softmax(read_logits, dim=-1)
        joint = self._circ_conv(read_p)                      # [B,C] injected mod-C combine
        return joint, read_logits


# ---------------------------------------------------------------------------
# Batch sampler (training). Enforces the WHOLE-DIAL global-H + aux-no-held assertions (spec §5'.3/§5'.4).
# ---------------------------------------------------------------------------
@dataclass
class Cfg:
    n_causes: int = 3
    C: int = 5
    n_train_worlds: int = 64
    n_test_worlds: int = 16
    pool: int = 2000
    n_obs: int = 400
    n_do: int = 400
    d: int = 64
    layers: int = 2
    heads: int = 4
    steps: int = 1500
    batch: int = 16
    lr: float = 3e-3
    aux_weight: float = 0.0     # 0 = aux-OFF (HEADLINE); >0 = aux-ON (CEILING / fairness control)
    p_skill: float = 0.5        # prob of whole-dial skill withholding per training episode (§5'.3.4)
    seed: int = 20260605
    device: str = "cuda"


class TrainSet:
    def __init__(self, cfg: Cfg, world_seeds, seed_off: int):
        self.cfg = cfg
        self.worlds, self.pools, self.queries, self.H = {}, {}, {}, {}
        for ws in world_seeds:
            w = make_world(cfg.n_causes, cfg.C, ws)
            self.worlds[ws] = w
            self.pools[ws] = WorldPool(w, cfg.pool, seed_off)
            self.H[ws] = holdout_for(w)            # WHOLE DIAL i*
            self.queries[ws] = valid_queries(w)
        self.world_seeds = list(world_seeds)

    def train_queries(self, ws):
        """Training queries HOLD the held dial at REF (spec §5'.3.2) -> no label depends on g_{i*}[non-REF]."""
        i_s = self.H[ws]
        return [a for a in self.queries[ws] if a[i_s] == REF]

    def held_queries(self, ws):
        """PRIMARY whole-dial transport test queries: those setting the held dial to a NON-REF value."""
        i_s = self.H[ws]
        return [a for a in self.queries[ws] if a[i_s] != REF]

    def whole_dial_omit(self, i_dial):
        return {(i_dial, v) for v in range(self.cfg.C)}

    def sample_batch(self, rng, no_forced=False):
        cfg = self.cfg
        rows_b, q_b, tgt_b, aux_b, auxm_b = [], [], [], [], []
        for _ in range(cfg.batch):
            ws = self.world_seeds[rng.integers(0, len(self.world_seeds))]
            w, pool = self.worlds[ws], self.pools[ws]
            tq = self.train_queries(ws)
            a = tq[rng.integers(0, len(tq))]
            i_s = self.H[ws]
            # global-H WHOLE DIAL is ALWAYS omitted (spec §5'.3). Skill training (§5'.3.4) ALSO withholds an
            # entire random non-i* dial used at non-REF in the query -> teaches whole-dial in-context inference.
            omit = self.whole_dial_omit(i_s)
            if rng.random() < cfg.p_skill:
                cand = [j for j in range(cfg.n_causes) if j != i_s and a[j] != REF]
                if cand:
                    omit |= self.whole_dial_omit(cand[rng.integers(0, len(cand))])
            n_do = 0 if no_forced else cfg.n_do
            arr, q = build_episode(pool, a, cfg.n_obs, n_do, omit, rng)
            # ASSERTION (spec §5'.3/§5'.4): NO clean forced block of the held dial i* ever appears in training.
            if not no_forced:
                ftags = set(arr[arr[:, FEAT_TYPE] == 1][:, FEAT_FDIAL].tolist())
                assert i_s not in ftags, f"LEAK: held dial i*={i_s} shown clean in train world {ws}"
            # aux targets g_p[a_p]; aux MASK supervises ONLY dials whose clean block is PRESENT. A dial whose
            # whole block is omitted (the global i* AND any per-episode skill-withheld dial) is left for the
            # model to INFER -> this is the curriculum fix: previously aux supervised the skill-withheld dial,
            # short-circuiting the very inference skill the skill episodes were meant to teach (spec flaw,
            # 2026-06-05). Deriving the mask from `omit` keeps the §5'.4 cell-identity guarantee (i* never aux'd).
            at = aux_targets(w, a)
            fully_withheld = {d for d in range(cfg.n_causes)
                              if all((d, v) in omit for v in range(cfg.C))}
            am = np.array([0.0 if p in fully_withheld else 1.0 for p in range(cfg.n_causes)], dtype=np.float32)
            assert am[i_s] == 0.0, f"AUX LEAK: aux target touches held dial i*={i_s}"
            rows_b.append(arr); q_b.append(q); tgt_b.append(exact_target(w, a))
            aux_b.append(at); auxm_b.append(am)
        R, mask = pad_batch(rows_b)
        return (torch.from_numpy(R), torch.from_numpy(np.stack(q_b)),
                torch.from_numpy(np.stack(tgt_b)).float(),
                torch.from_numpy(np.stack(aux_b)), torch.from_numpy(np.stack(auxm_b)),
                torch.from_numpy(mask))


def kl_loss(joint, target):
    logp = torch.log(joint.clamp_min(1e-9))
    return F.kl_div(logp, target, reduction="batchmean")


def aux_loss(read_logits, aux_tgt, aux_mask):
    """Masked CE of per-dial reads vs g_p[a_p]; mask zeros the held dial (§5'.4)."""
    B, nC, C = read_logits.shape
    ce = F.cross_entropy(read_logits.reshape(B * nC, C), aux_tgt.reshape(B * nC), reduction="none")
    m = aux_mask.reshape(B * nC)
    return (ce * m).sum() / m.sum().clamp_min(1.0)


def train_model(ts: TrainSet, cfg: Cfg, no_forced=False, tag="main") -> Rung1Model:
    dev = cfg.device if torch.cuda.is_available() else "cpu"
    model = Rung1Model(cfg.n_causes, cfg.C, cfg.d, cfg.heads, cfg.layers).to(dev)
    n_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
    rng = np.random.default_rng(cfg.seed + (777 if no_forced else 0) + int(cfg.aux_weight > 0) * 333)
    t0 = time.time()
    model.train()
    for step in range(cfg.steps):
        rows, q, tgt, at, am, pm = ts.sample_batch(rng, no_forced=no_forced)
        rows, q, tgt, at, am, pm = (rows.to(dev), q.to(dev), tgt.to(dev), at.to(dev), am.to(dev), pm.to(dev))
        joint, read_logits = model(rows, q, pad_mask=pm)
        loss = kl_loss(joint, tgt)
        if cfg.aux_weight > 0:
            loss = loss + cfg.aux_weight * aux_loss(read_logits, at, am)
        opt.zero_grad(); loss.backward(); opt.step()
        if step % max(1, cfg.steps // 8) == 0 or step == cfg.steps - 1:
            print(f"  [{tag}] step {step:>5}/{cfg.steps}  loss={loss.item():.4f}  "
                  f"({time.time()-t0:.0f}s, {n_params/1e3:.0f}k params)", flush=True)
    return model


# ---------------------------------------------------------------------------
# Eval: build episodes for a grid cell, score model & controls vs sampled oracle (eval parity).
# ---------------------------------------------------------------------------
@torch.no_grad()
def model_pred(model, rows, q, pad_mask=None):
    dev = next(model.parameters()).device
    model.eval()
    pm = pad_mask.to(dev) if pad_mask is not None else None
    joint, _ = model(rows.to(dev), q.to(dev), pad_mask=pm)
    return joint.cpu().numpy()


def _apply_shuffle(arr, nC, rng):
    """STRENGTHENED forced<->query shuffle (spec §B'): consistently relabel the episode's dial axis (BOTH
    the dial value columns AND the forced-dial tags) by a non-identity permutation, while the QUERY stays
    fixed. The episode remains an internally-consistent (relabeled) world, but query slot p now points at a
    different underlying dial -> a genuine reader provably drops to floor. (The old shuffle permuted only the
    FEAT_FDIAL tag, leaving the constant dial COLUMN as an unbroken cue a model could read instead.)"""
    perm = rng.permutation(nC)
    while np.array_equal(perm, np.arange(nC)):
        perm = rng.permutation(nC)
    inv = np.argsort(perm)
    arr = arr.copy()
    arr[:, 3:3 + nC] = arr[:, 3:3 + nC][:, perm]            # new col j <- old col perm[j]
    fmask = arr[:, FEAT_TYPE] == 1
    arr[fmask, FEAT_FDIAL] = inv[arr[fmask, FEAT_FDIAL]]    # tag d -> slot now holding dial d
    return arr


def eval_cell(model, ts: TrainSet, cfg: Cfg, world_seeds, withheld: bool, shuffle=False,
              rng_seed=12345) -> dict:
    """Per-world mean model score + cheap analytic controls, over queries in the cell."""
    rng = np.random.default_rng(rng_seed)
    per_world_model, per_world_ctrl = [], {b: [] for b in
                                           ("one_cause", "modal_obs", "nearest_cell", "retrieval")}
    min_q = 1e9
    for ws in world_seeds:
        w, pool = ts.worlds[ws], ts.pools[ws]
        i_s = ts.H[ws]
        # BOTH cells use the SAME held queries (a[i*]!=REF) -> a clean controlled contrast that toggles ONLY
        # the withholding. withheld: dial i*'s whole block omitted (must infer cross-block = transport).
        # revealed: ALL blocks present (true all-revealed sanity = can the model read+compose at all?).
        qs = ts.held_queries(ws)
        if not qs:
            continue
        min_q = min(min_q, len(qs))
        omit = ts.whole_dial_omit(i_s) if withheld else set()
        m_scores, c_scores = [], {b: [] for b in per_world_ctrl}
        rows_b, q_b, oracles_b, assigns = [], [], [], []
        for a in qs:
            arr, q = build_episode(pool, a, cfg.n_obs, cfg.n_do, omit, rng)
            if shuffle:
                arr = _apply_shuffle(arr, cfg.n_causes, rng)
            rows_b.append(arr); q_b.append(q); assigns.append(a)
            oracles_b.append(oracles(w, a))
        R, mask = pad_batch(rows_b)
        preds = model_pred(model, torch.from_numpy(R), torch.from_numpy(np.stack(q_b)),
                           pad_mask=torch.from_numpy(mask))
        for k, a in enumerate(assigns):
            orc = oracles_b[k]
            m_scores.append(norm_score(preds[k], orc["oracle"]))
            for b in ("one_cause", "modal_obs", "nearest_cell"):
                c_scores[b].append(norm_score(orc[b], orc["oracle"]))
            c_scores["retrieval"].append(norm_score(_retrieval_pred(pool, a, cfg, rng), orc["oracle"]))
        per_world_model.append(float(np.mean(m_scores)))
        for b in per_world_ctrl:
            per_world_ctrl[b].append(float(np.mean(c_scores[b])))
    return {"model_mean": float(np.mean(per_world_model)),
            "model_ci": _world_boot(per_world_model),
            "controls": {b: float(np.mean(v)) for b, v in per_world_ctrl.items()},
            "controls_ci": {b: _world_boot(v) for b, v in per_world_ctrl.items()},
            "per_world_model": per_world_model, "min_queries": int(min_q)}


def _retrieval_pred(pool: WorldPool, assign, cfg: Cfg, rng) -> np.ndarray:
    """[CDX-6] naive retrieval+add: marginal modal R per single-do(dial=a_i) (NO deconfounding), sum modC.
    NOTE: for the held dial i* the clean block is absent, so this control falls back to its observational
    column mode — exactly the elimination/lookup route the whole-dial holdout is designed to deny."""
    C, nC = cfg.C, cfg.n_causes
    i_s = holdout_for(pool.world)
    s = 0
    for i in range(nC):
        if (i, assign[i]) in pool.forced and i != i_s:
            _, R = pool.sample_forced(i, assign[i], cfg.n_do, rng)
        else:  # held dial: no clean block -> use observational rows where that dial == assign[i]
            od, oR = pool.obs
            mask = od[:, i] == assign[i]
            R = oR[mask] if mask.any() else oR
        s += int(np.bincount(R, minlength=C).argmax())
    p = np.zeros(C); p[s % C] = 1.0
    return p


def _world_boot(vals, reps=10000, seed=0):
    vals = np.asarray([v for v in vals if not np.isnan(v)], float)
    if len(vals) == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    m = vals[rng.integers(0, len(vals), size=(reps, len(vals)))].mean(1)
    return (float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5)))


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", action="store_true", help="pre-registered budget (800 rows/block @3 dials)")
    ap.add_argument("--seed", type=int, default=20260605)
    ap.add_argument("--steps", type=int)
    ap.add_argument("--d", type=int); ap.add_argument("--layers", type=int); ap.add_argument("--heads", type=int)
    ap.add_argument("--lr", type=float); ap.add_argument("--n-do", type=int); ap.add_argument("--n-obs", type=int)
    ap.add_argument("--train-worlds", type=int); ap.add_argument("--pool", type=int)
    ap.add_argument("--aux-weight", type=float, default=0.5, help="aux-ON ceiling weight (aux-OFF arm is 0)")
    ap.add_argument("--only-auxon", action="store_true",
                    help="train ONLY the curriculum-fixed aux-ON ceiling + eval its withheld/revealed/shuffle "
                         "(diagnostic: aux-OFF/no-forced are unaffected by the aux-mask fix)")
    ap.add_argument("--eval-worlds", type=int, default=0, help="cap eval worlds per cell (0 = all)")
    ap.add_argument("--out-json", type=Path,
                    default=Path(__file__).resolve().parent.parent / "results" / "RUNG1_RESULT.json")
    args = ap.parse_args()

    cfg = Cfg(seed=args.seed)
    if args.full:
        cfg.n_obs, cfg.n_do, cfg.pool = 800, 800, 3000
        cfg.n_train_worlds, cfg.n_test_worlds = 96, 24
        cfg.steps = 3000
    for k in ("steps", "d", "layers", "heads", "lr", "pool"):
        if getattr(args, k) is not None:
            setattr(cfg, k, getattr(args, k))
    if args.n_do is not None: cfg.n_do = args.n_do
    if args.n_obs is not None: cfg.n_obs = args.n_obs
    if args.train_worlds is not None: cfg.n_train_worlds = args.train_worlds

    print("=" * 96)
    print(f"RUNG-1.5 {'FULL' if args.full else 'SMOKE'} (conv+aux, WHOLE-DIAL) — {cfg.n_causes} dials "
          f"C={cfg.C} | n_obs={cfg.n_obs} n_do={cfg.n_do} | {cfg.n_train_worlds} train / "
          f"{cfg.n_test_worlds} test worlds | steps={cfg.steps} | seed={cfg.seed}")
    print("=" * 96)

    base = np.random.default_rng(cfg.seed)
    seeds = base.integers(1, 2**31 - 1, size=cfg.n_train_worlds + cfg.n_test_worlds).tolist()
    train_seeds, test_seeds = seeds[:cfg.n_train_worlds], seeds[cfg.n_train_worlds:]
    ts = TrainSet(cfg, train_seeds + test_seeds, seed_off=1)
    ts.world_seeds = list(train_seeds)  # training samples from train worlds only

    cap = args.eval_worlds or 10**9
    train_eval = train_seeds[:cap]

    # ---- Diagnostic: curriculum-fixed aux-ON only (2026-06-05). Tests whether in-context whole-dial
    # inference is learnable WITH read-help once aux no longer supervises the skill-withheld dial. ----
    if args.only_auxon:
        on_cfg = Cfg(**{**asdict(cfg), "aux_weight": args.aux_weight})
        m = train_model(ts, on_cfg, no_forced=False, tag="aux-ON(curr-fixed)")
        wh = eval_cell(m, ts, cfg, train_eval, withheld=True, rng_seed=cfg.seed + 13)
        rv = eval_cell(m, ts, cfg, train_eval, withheld=False, rng_seed=cfg.seed + 15)
        shf = eval_cell(m, ts, cfg, train_eval, withheld=True, shuffle=True, rng_seed=cfg.seed + 9)
        thr = 0.80 * CEILING
        sc = wh["controls"]; sn = max(sc, key=sc.get)
        print("\n--- CURRICULUM-FIXED aux-ON (must INFER every withheld dial; others supervised) ---")
        print(f"  ceiling {CEILING:+.3f}  bar 0.80*ceiling={thr:+.3f}")
        print(f"  withheld (PRIMARY) {wh['model_mean']:+.3f} CI[{wh['model_ci'][0]:+.2f},{wh['model_ci'][1]:+.2f}]")
        print(f"  revealed (present) {rv['model_mean']:+.3f} CI[{rv['model_ci'][0]:+.2f},{rv['model_ci'][1]:+.2f}]")
        print(f"  +shuffle           {shf['model_mean']:+.3f}  strongest_ctrl={sc[sn]:+.3f} ({sn})")
        learns_inference = wh["model_mean"] >= thr
        print(f"\n>>> aux-ON-fixed verdict: "
              f"{'INFERENCE LEARNABLE WITH READ-HELP (transport learnable; from-scratch is the open Q)' if learns_inference else 'INFERENCE NOT LEARNABLE EVEN WITH READ-HELP (strong NULL)'}")
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(json.dumps(
            {"cfg": asdict(on_cfg), "ceiling": CEILING, "withheld": wh, "revealed": rv, "shuffle": shf,
             "learns_inference_with_help": bool(learns_inference)},
            default=lambda o: o.tolist() if isinstance(o, np.ndarray) else list(o)
            if isinstance(o, tuple) else o, indent=2) + "\n")
        print(f"Wrote {args.out_json}")
        return

    # ---- Train the three pre-registered models. aux-OFF is the HEADLINE; aux-ON is the ceiling control. ----
    off_cfg = Cfg(**{**asdict(cfg), "aux_weight": 0.0})
    main_model = train_model(ts, off_cfg, no_forced=False, tag="aux-OFF(headline)")
    noforced_model = train_model(ts, off_cfg, no_forced=True, tag="no-forced")
    on_cfg = Cfg(**{**asdict(cfg), "aux_weight": args.aux_weight})
    auxon_model = train_model(ts, on_cfg, no_forced=False, tag="aux-ON(ceiling)")

    cap = args.eval_worlds or 10**9
    train_eval, test_eval = train_seeds[:cap], test_seeds[:cap]
    print("\n--- EVAL GRID — aux-OFF headline (per-world means, 95% bootstrap-by-world) ---")
    cells = {}
    for wname, wseeds in (("seen", train_eval), ("unseen", test_eval)):
        for withheld in (False, True):
            key = f"{wname}/{'withheld' if withheld else 'revealed'}"
            r = eval_cell(main_model, ts, cfg, wseeds, withheld=withheld, rng_seed=cfg.seed + 5)
            cells[key] = r
            cl, ch = r["model_ci"]
            strongest = max(r["controls"].values())
            print(f"  {key:<18} model={r['model_mean']:+.3f}[{cl:+.2f},{ch:+.2f}]  "
                  f"strongest_ctrl={strongest:+.3f}  (min_q/world={r['min_queries']})")

    # ---- Mandatory guards on the PRIMARY cell (seen / whole-dial withheld) ----
    sh = eval_cell(main_model, ts, cfg, train_eval, withheld=True, shuffle=True, rng_seed=cfg.seed + 9)
    cells["seen/withheld+shuffle"] = sh
    nf = eval_cell(noforced_model, ts, cfg, train_eval, withheld=True, rng_seed=cfg.seed + 11)
    cells["seen/withheld_noforced"] = nf
    on = eval_cell(auxon_model, ts, cfg, train_eval, withheld=True, rng_seed=cfg.seed + 13)
    cells["seen/withheld_auxON"] = on
    on_rev = eval_cell(auxon_model, ts, cfg, train_eval, withheld=False, rng_seed=cfg.seed + 15)
    cells["seen/revealed_auxON"] = on_rev

    prim = cells["seen/withheld"]
    strongest_name = max(prim["controls"], key=prim["controls"].get)
    strongest_ctrl = prim["controls"][strongest_name]
    strongest_ci = prim["controls_ci"][strongest_name]
    thr = 0.80 * CEILING
    floor = strongest_ctrl
    print("\n--- PRIMARY CELL: seen-world / WHOLE-DIAL withheld-effect (aux-OFF headline) ---")
    print(f"  ceiling (analytic, §5'.6)  {CEILING:+.3f}   PASS bar 0.80*ceiling = {thr:+.3f}")
    print(f"  model (aux-OFF)  {prim['model_mean']:+.3f} CI[{prim['model_ci'][0]:+.2f},{prim['model_ci'][1]:+.2f}]")
    print(f"  strongest ctrl   {strongest_ctrl:+.3f} ({strongest_name}) CI[{strongest_ci[0]:+.2f},{strongest_ci[1]:+.2f}]")
    print(f"  +shuffle         {sh['model_mean']:+.3f} CI[{sh['model_ci'][0]:+.2f},{sh['model_ci'][1]:+.2f}]  (must drop <= floor+0.10)")
    print(f"  no-forced model  {nf['model_mean']:+.3f} CI[{nf['model_ci'][0]:+.2f},{nf['model_ci'][1]:+.2f}]  (honest ablation)")
    print(f"  aux-ON ceiling   {on['model_mean']:+.3f} CI[{on['model_ci'][0]:+.2f},{on['model_ci'][1]:+.2f}]  (NOT transport evidence)")
    print(f"  sanity revealed  {cells['seen/revealed']['model_mean']:+.3f}  (aux-OFF all-revealed read)")

    # ---- Pre-registered ceiling-relative verdict (spec §B') ----
    beats = prim["model_ci"][0] > strongest_ci[1]
    shuffle_drops = sh["model_mean"] <= floor + 0.10
    approaches = prim["model_mean"] >= thr
    revealed_ok = cells["seen/revealed"]["model_mean"] >= thr
    if approaches and beats and shuffle_drops:
        verdict = "TRANSPORTS"
    elif prim["model_mean"] <= floor + 0.15 and revealed_ok:
        verdict = "COLLAPSE"
    elif not revealed_ok:
        verdict = "CAPACITY/OPTIMIZATION_NULL"   # can't learn even the all-revealed read at this capacity
    else:
        verdict = "INCONCLUSIVE"
    print(f"\n>>> PRIMARY verdict (this run, single seed): {verdict}")
    print(f"    approaches_ceiling(>= {thr:+.2f})={approaches}  beats_strongest={beats}  "
          f"shuffle_drops={shuffle_drops}  all_revealed_ok={revealed_ok}")
    print("    [single-seed; pre-registered PASS needs all 3 seeds + the §5'.5 recovery-geometry control — spec §B']")

    payload = {"cfg": asdict(cfg), "ceiling": CEILING, "pass_bar": thr, "verdict": verdict,
               "guards": {"beats_strongest": beats, "shuffle_drops": shuffle_drops,
                          "approaches_ceiling": approaches, "all_revealed_ok": revealed_ok},
               "cells": cells}
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(payload,
                             default=lambda o: o.tolist() if isinstance(o, np.ndarray) else list(o)
                             if isinstance(o, tuple) else o, indent=2) + "\n")
    print(f"Wrote {args.out_json}")


if __name__ == "__main__":
    main()
