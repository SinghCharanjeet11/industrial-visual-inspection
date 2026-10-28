"""Check how much recall the gray bank keeps: score the COLOUR test images of a category, converted to gray, with
the gray bank, and compare with the colour-bank decisions (pseudo-labels; no ground truth used).
Writes <out>/<cat>_graycheck_<backbone>.npz (ids, gray-bank scores of colour test images).
Usage: python gray_check.py <data> <out> <cat> [backbones]
"""
import os, sys, time
import numpy as np, pandas as pd
from PIL import Image
import prep
from pc2 import CFG, preprocess
from pc3 import BACKBONES, Extractor, Bank

data, out, cat = sys.argv[1:4]
bbs = (sys.argv[4] if len(sys.argv) > 4 else "wrn50,dn201").split(",")
cfg = dict(CFG[cat])
tr, te = pd.read_csv(os.path.join(data, "train.csv")), pd.read_csv(os.path.join(data, "test.csv"))
clips = os.path.join(data, "clips")
t0 = time.time()
gimgs = [prep.to_gray(preprocess(Image.open(os.path.join(clips, p)).convert("RGB"), cat, cfg["size"])[0])
         for p in tr[tr.category == cat].path]
tdf = te[te.category == cat].reset_index(drop=True)
tpre = [preprocess(Image.open(os.path.join(clips, p)).convert("RGB"), cat, cfg["size"]) for p in tdf.path]
sel = [j for j, (_, g) in enumerate(tpre) if not g]
timgs = [prep.to_gray(tpre[j][0]) for j in sel]
for m in bbs:
    bank = Bank(Extractor(BACKBONES[m]()).eval(), gimgs, cfg)
    np.savez(os.path.join(out, f"{cat}_graycheck_{m}.npz"), test_ids=tdf.id.values[sel], test=bank.score(timgs))
    print(f"{m} done ({time.time() - t0:.0f}s)", flush=True)
