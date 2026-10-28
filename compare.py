"""Compare backbone combinations for one (category, bank) using unsupervised test-score diagnostics.
  synAUROC - good (leave-one-out) vs synthetic defects
  sep      - 2-GMM separation of test z-scores
  amb      - number of test images within +-1 z of the p97.5 threshold (fewer = cleaner decision)
  flag     - fraction flagged at p97.5
Usage: python compare.py <out_dir> <cat> <bank> model1 model2 ... [--max 3]
"""
import itertools, os, sys
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.mixture import GaussianMixture

args = [x for x in sys.argv[1:] if not x.startswith("--")]
out, cat, bank, models = args[0], args[1], args[2], args[3:]
mx = int(sys.argv[sys.argv.index("--max") + 1]) if "--max" in sys.argv else len(models)
if "--max" in sys.argv:
    models = [m for m in models if m != str(mx)]
z = {}
for m in models:
    d = np.load(os.path.join(out, f"{cat}_{bank}_{m}.npz"), allow_pickle=True)
    mu, sd = d["good"].mean(), d["good"].std()
    z[m] = {k: (d[k] - mu) / sd for k in ("test", "good", "syn")}
res = []
for r in range(1, mx + 1):
    for ms in itertools.combinations(models, r):
        e = {k: np.mean([z[m][k] for m in ms], 0) for k in ("test", "good", "syn")}
        auc = roc_auc_score(np.r_[0 * e["good"], 1 + 0 * e["syn"]], np.r_[e["good"], e["syn"]])
        t = e["test"]; th = np.percentile(e["good"], 97.5)
        gm = GaussianMixture(2, random_state=0).fit(t.reshape(-1, 1))
        mu, sd = gm.means_.ravel(), np.sqrt(gm.covariances_.ravel())
        res.append(("+".join(ms), auc, abs(mu[0] - mu[1]) / (sd[0] + sd[1]), int(((t > th - 1) & (t < th + 1)).sum()), (t > th).mean()))
for n, auc, sep, amb, fl in sorted(res, key=lambda r: r[3]):
    print(f"{n:40s} synAUROC={auc:.4f} sep={sep:.2f} amb={amb:3d} flag={fl:.2f}")
