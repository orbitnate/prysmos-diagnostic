"""Council killshot test (Model Council, 2026-06-05): does a NON-compositional
'retrieval + mod-C add' strategy ALSO clear the bar, tying the real composer?

Their claim: our 5 cheaters fail only because they DISCARD the forced data; a smart
shortcut that READS the revealed dial effects and adds them mod C would score ~0.94,
proving the tool cannot tell composition from retrieval+arithmetic.

We test two versions, $0:
  (A) NAIVE retrieval+add  -- marginal mode of R under each single-do (other dials FREE,
      i.e. still confounded), relative-shift + anchor, sum mod C. This is the genuinely
      'dumb' lookup the council describes -- no deconfounding.
  (B) DECONFOUNDED composer -- the existing analytic_composer (conditions on other dials
      == reference to cancel the shared hidden background before reading each effect).

If (A) ~ (B) ~ 0.94  -> council is RIGHT, our cheater separation has a hole.
If (A) ~ 0  and (B) ~ 0.94 -> the ONLY strategy that passes must deconfound-then-compose;
   no dumb retrieval clears the bar, so the discriminant separation HOLDS and only the
   CLAIM WORDING (asserting internal mechanism) needs fixing.
"""
import numpy as np
from llm_composition_ramp import (make_world, simulate, oracles, norm_score,
                                  analytic_composer, gate_level, true_value, _bootstrap_ci)

SEEDS = [20260605, 20260606, 20260607]
LEVELS = [2, 3, 4]
C = 5
N = 40000


def naive_retrieval(world, assign, n=N):
    """(A) Dumb lookup+add: marginal mode of R|do(dial_i=v) with OTHER dials free (confounded).
    Relative shift from reference r=0, single marginal anchor, sum mod C. NO conditioning."""
    nC, Cc = world["n"], world["C"]
    rng = np.random.default_rng(world["seed"] * 104729 + 99)
    r = 0

    def marg_mode(i, v):
        _, R = simulate(world, n, rng, do={i: v})  # other dials roam with hidden H (confounded)
        return int(np.bincount(R, minlength=Cc).argmax())

    base = [marg_mode(i, r) for i in range(nC)]
    deltas = np.zeros((nC, Cc), dtype=int)
    for i in range(nC):
        for v in range(Cc):
            deltas[i, v] = (marg_mode(i, v) - base[i]) % Cc
    K = base[0]  # one marginal anchor (contains one background, like a memorizer would grab)
    pred_val = (K + sum(int(deltas[i, assign[i]]) for i in range(nC))) % Cc
    p = np.zeros(Cc); p[pred_val] = 1.0
    return p


def main():
    print(f"{'seed':>9} {'level':>5} | {'NAIVE retrieval+add':>22} | {'DECONFOUND composer':>22}")
    print("-" * 64)
    allnaive, allcomp = [], []
    for seed in SEEDS:
        for L in LEVELS:
            world = make_world(L, C, seed)
            pairs = gate_level(L, C, seed, 30)["pairs"]
            orac = {a: oracles(world, a)["oracle"] for a in pairs}
            ns, cs = [], []
            for a in pairs:
                ns.append(norm_score(naive_retrieval(world, a), orac[a]))
                cp = analytic_composer(world, a)
                cs.append(norm_score(cp, orac[a]) if cp is not None else float("nan"))
            ns, cs = np.array(ns), np.array(cs)
            nlo, nhi = _bootstrap_ci(ns); clo, chi = _bootstrap_ci(cs)
            allnaive += list(ns); allcomp += list(cs)
            print(f"{seed:>9} {L:>5} | {np.nanmean(ns):+.3f} [{nlo:+.2f},{nhi:+.2f}] "
                  f"| {np.nanmean(cs):+.3f} [{clo:+.2f},{chi:+.2f}]")
    an, ac = np.array(allnaive), np.array(allcomp)
    print("-" * 64)
    nlo, nhi = _bootstrap_ci(an); clo, chi = _bootstrap_ci(ac)
    print(f"{'POOLED':>15} | {np.nanmean(an):+.3f} [{nlo:+.2f},{nhi:+.2f}] "
          f"| {np.nanmean(ac):+.3f} [{clo:+.2f},{chi:+.2f}]")
    print()
    naive_passes = np.nanmean(an) >= 0.90
    print("VERDICT:")
    if naive_passes:
        print("  Naive retrieval+add CLEARS 0.90 -> council RIGHT: tool cannot separate")
        print("  retrieval from composition. Cheater set has a real hole.")
    else:
        print(f"  Naive retrieval+add = {np.nanmean(an):+.3f} (FAILS the >=0.90 bar);")
        print(f"  only the deconfound-then-compose strategy ({np.nanmean(ac):+.3f}) passes.")
        print("  => No DUMB retrieval clears the bar. Discriminant separation HOLDS;")
        print("     fix is claim WORDING (don't assert internal mechanism), not the tool.")


if __name__ == "__main__":
    main()
