"""Unsupervised diagnostics of pc3 test-score distributions (no labels used).

For each (category, bank) and each backbone / the ensemble, standardise by leave-one-out good scores, then report:
  sep   - 2-component GMM separation on test z-scores: |mu1-mu2| / (sd1+sd2)  (higher = cleaner split)
  gap   - widest gap between consecutive sorted test z-scores within [lo, hi] (where the threshold should sit)
  flag  - fraction flagged at LOO-good p97.5
Usage: python inspect_scores.py <out_dir> [cat ...] [--show]
"""
import sys, os
import numpy as np
from sklearn.mixture import GaussianMixture

out = sys.argv[1]
cats = [c for c in sys.argv[2:] if not c.startswith("--")] or ["bottle", "cable", "hazelnut", "screw"]
show = "--show" in sys.argv
LO, HI = 0.0, 8.0


def diag(t, g):
    gm = GaussianMixture(2, random_state=0).fit(t.reshape(-1, 1))
    mu, sd = gm.means_.ravel(), np.sqrt(gm.covariances_.ravel())
    sep = abs(mu[0] - mu[1]) / (sd[0] + sd[1])
    s = np.sort(t); m = (s[:-1] >= LO) & (s[1:] <= HI)
    gaps = np.diff(s)[m]
    k = int(np.argmax(gaps)) if gaps.size else None
    gap = (gaps[k], s[:-1][m][k], s[1:][m][k], (s > s[:-1][m][k]).mean()) if k is not None else (0, 0, 0, 0)
    return sep, gap, (t > np.percentile(g, 97.5)).mean()


for cat in cats:
    for b in ["color", "gray"]:
        fs = {m: os.path.join(out, f"{cat}_{b}_{m}.npz") for m in ["wrn50", "dn201"]}
        fs = {m: f for m, f in fs.items() if os.path.exists(f)}
        if not fs:
            continue
        z = {}
        for m, f in fs.items():
            d = np.load(f); mu, sd = d["good"].mean(), d["good"].std()
            z[m] = ((d["test"] - mu) / sd, (d["good"] - mu) / sd)
        if len(z) == 2:
            z["ens"] = tuple((z["wrn50"][i] + z["dn201"][i]) / 2 for i in range(2))
        for m, (t, g) in z.items():
            sep, (gw, ga, gb, gf), fl = diag(t, g)
            print(f"{cat:9s}{b:6s}{m:6s} n={len(t):3d} sep={sep:.2f}  widest gap {gw:.2f} in ({ga:.1f},{gb:.1f}) "
                  f"-> flag {gf:.2f} | LOO p97.5 thr={np.percentile(g, 97.5):.2f} flag {fl:.2f}")
            if show:
                print("   ", " ".join(f"{x:.1f}" for x in np.sort(t)))
