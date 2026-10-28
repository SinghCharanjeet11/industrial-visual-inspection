"""PatchCore v2 for DefectVision (F1, defective = positive).

Per category:
  - NLM denoise every image (train and test identically)
  - screw: grayscale + pose canonicalization (horizontal, head left) + rectangular crop
  - separate color / gray memory banks (gray bank = grayscale copies of all train goods)
  - optional rotation augmentation of the bank (bottle, hazelnut)
  - cable: position-aware nearest neighbour (patch only matched to nearby positions)
K-fold CV on train goods + synthetic on-object defects -> proxy AUROC / F1; threshold per (category, bank)
chosen as a percentile of out-of-fold good scores.
"""
import argparse, json, math, os, random, time

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torchvision
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter
from scipy.ndimage import gaussian_filter
from sklearn.metrics import roc_auc_score

import prep

torch.set_num_threads(12)
EMB, PROJ = 512, 128
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

CFG = {
    "bottle":   dict(size=(256, 256), rot=True,  pos=False, coreset=12000),
    "cable":    dict(size=(224, 224), rot=False, pos=True,  coreset=12000, radius=2),
    "hazelnut": dict(size=(256, 256), rot=True,  pos=False, coreset=12000),
    "screw":    dict(size=(448, 160), rot=False, pos=False, coreset=12000),
}
PCTS = [90, 95, 97.5, 99, 100]


class Extractor(torch.nn.Module):
    def __init__(self):
        super().__init__()
        m = torchvision.models.wide_resnet50_2(weights="IMAGENET1K_V1").eval()
        self.stem = torch.nn.Sequential(m.conv1, m.bn1, m.relu, m.maxpool, m.layer1)
        self.l2, self.l3 = m.layer2, m.layer3

    @torch.no_grad()
    def forward(self, x):
        x = (x - MEAN) / STD
        f2 = self.l2(self.stem(x))
        f3 = self.l3(f2)
        f2 = F.avg_pool2d(f2, 3, 1, 1)
        f3 = F.avg_pool2d(f3, 3, 1, 1)
        f3 = F.interpolate(f3, size=f2.shape[-2:], mode="bilinear", align_corners=False)
        f = torch.cat([f2, f3], 1)
        B, C, H, W = f.shape
        f = f.permute(0, 2, 3, 1).reshape(-1, 1, C)
        return F.adaptive_avg_pool1d(f, EMB).reshape(B, H, W, EMB)


# ---------------- preprocessing ----------------
def preprocess(im, cat, size):
    im = prep.denoise(im)
    if cat == "screw":
        im = prep.to_gray(im)
        return prep.screw_align(im, size), True
    g = prep.is_gray(im)
    if g:
        im = prep.to_gray(im)
    return im.resize(size, Image.BILINEAR), g


def to_tensor(imgs):
    arr = np.stack([np.asarray(im, dtype=np.float32) / 255.0 for im in imgs])
    return torch.from_numpy(arr).permute(0, 3, 1, 2)


def embed(model, imgs, bs=8):
    out = []
    for i in range(0, len(imgs), bs):
        out.append(model(to_tensor(imgs[i:i + bs])).half())
    return torch.cat(out)  # N,H,W,D fp16


def coreset(bank, k):
    n = bank.shape[0]
    k = min(k, n)
    g = torch.Generator().manual_seed(0)
    proj = torch.randn(bank.shape[1], PROJ, generator=g) / math.sqrt(PROJ)
    z = torch.empty(n, PROJ)
    for i in range(0, n, 65536):
        z[i:i + 65536] = bank[i:i + 65536].float() @ proj
    zn = (z * z).sum(1)
    sel = [0]
    mind = zn - 2 * (z @ z[0]) + zn[0]
    for _ in range(k - 1):
        j = int(torch.argmax(mind))
        sel.append(j)
        torch.minimum(mind, zn - 2 * torch.mv(z, z[j]) + zn[j], out=mind)
    return bank[torch.tensor(sel)].float()


def rot_aug(imgs):
    return imgs + [im.transpose(t) for im in imgs for t in (Image.ROTATE_90, Image.ROTATE_180, Image.ROTATE_270)]


class Bank:
    def __init__(self, model, imgs, cfg):
        self.model, self.cfg = model, cfg
        src = rot_aug(imgs) if cfg["rot"] else imgs
        if cfg["pos"]:
            feats = embed(model, src)
            self.hw = feats.shape[1:3]
            N, H, W, D = feats.shape
            self.pos = feats.float().reshape(N, H * W, D).permute(1, 0, 2).contiguous()  # HW,N,D
            self.pos_n = (self.pos ** 2).sum(-1)                                         # HW,N
            del feats
            return
        # low-memory: coreset each chunk of images, then coreset the union
        K, CH = cfg["coreset"], 32
        nchunks = math.ceil(len(src) / CH)
        keep = []
        for c in range(nchunks):
            f = embed(model, src[c * CH:(c + 1) * CH])
            self.hw = f.shape[1:3]
            f = f.reshape(-1, EMB)
            keep.append(coreset(f, max(1, min(len(f), 2 * K // nchunks + 1))).half())
            del f
        self.mem = coreset(torch.cat(keep), K)

    def patch_dist(self, f):
        """f: H,W,D fp16 -> H,W nearest-neighbour distance."""
        H, W = self.hw
        if not self.cfg["pos"]:
            return torch.cdist(f.reshape(-1, EMB).float(), self.mem).min(1).values.reshape(H, W)
        r = self.cfg.get("radius", 2)
        t = f.float()
        best = torch.full((H, W), float("inf"))
        for dy in range(-r, r + 1):
            for dx in range(-r, r + 1):
                # compare test[h, w] with bank[:, h+dy, w+dx]: shift the test map instead of the bank
                ts = torch.zeros(H, W, EMB)
                ts[max(0, dy):H + min(0, dy), max(0, dx):W + min(0, dx)] = t[max(0, -dy):H - max(0, dy), max(0, -dx):W - max(0, dx)]
                tv = ts.reshape(H * W, EMB)
                cross = torch.bmm(self.pos, tv.unsqueeze(-1)).squeeze(-1)                # HW,N
                d2 = (self.pos_n - 2 * cross).min(1).values + (tv ** 2).sum(-1)
                d = d2.clamp_min(0).sqrt().reshape(H, W)
                # d indexed at bank position (h+dy, w+dx); move back to test position (h, w)
                best[max(0, -dy):H - max(0, dy), max(0, -dx):W - max(0, dx)] = torch.minimum(
                    best[max(0, -dy):H - max(0, dy), max(0, -dx):W - max(0, dx)],
                    d[max(0, dy):H + min(0, dy), max(0, dx):W + min(0, dx)])
        return best

    def score(self, imgs):
        feats = embed(self.model, imgs)
        out = []
        for f in feats:
            d = self.patch_dist(f).numpy()
            W, H = imgs[0].size
            amap = cv2.resize(d, (W, H), interpolation=cv2.INTER_LINEAR)
            out.append(float(gaussian_filter(amap, sigma=4).max()))
        return np.array(out)


# ---------------- synthetic on-object defects (validation proxy) ----------------
def obj_box(im, cat):
    w, h = im.size
    if cat == "screw":   # aligned: screw along the horizontal mid line
        return (0.12 * w, 0.40 * h, 0.88 * w, 0.60 * h)
    return (0.25 * w, 0.25 * h, 0.75 * w, 0.75 * h)


def _pt(box, rng):
    return rng.uniform(box[0], box[2]), rng.uniform(box[1], box[3])


def syn_cutpaste(im, cat, rng):
    w, h = im.size
    s = 0.04 if cat == "screw" else 0.10
    pw, ph = max(4, int(w * rng.uniform(s, 2 * s))), max(4, int(h * rng.uniform(s, 2 * s)))
    x0, y0 = rng.randint(0, w - pw), rng.randint(0, h - ph)
    patch = ImageEnhance.Brightness(im.crop((x0, y0, x0 + pw, y0 + ph))).enhance(rng.uniform(0.6, 1.4))
    patch = patch.rotate(rng.uniform(0, 360))
    cx, cy = _pt(obj_box(im, cat), rng)
    out = im.copy(); out.paste(patch, (int(cx - pw / 2), int(cy - ph / 2)))
    return out


def syn_scar(im, cat, rng):
    w, h = im.size
    out = im.copy(); d = ImageDraw.Draw(out)
    cx, cy = _pt(obj_box(im, cat), rng)
    L, a = rng.uniform(0.05, 0.15) * w, rng.uniform(0, math.pi)
    v = rng.randint(0, 255)
    col = (v, v, v) if cat == "screw" else tuple(rng.randint(0, 255) for _ in range(3))
    d.line([(cx - L / 2 * math.cos(a), cy - L / 2 * math.sin(a)), (cx + L / 2 * math.cos(a), cy + L / 2 * math.sin(a))],
           fill=col, width=max(2, w // 100))
    return out


def syn_blob(im, cat, rng):
    w, h = im.size
    out = im.copy()
    mask = Image.new("L", im.size, 0); d = ImageDraw.Draw(mask)
    r = rng.uniform(0.015, 0.03) * w if cat == "screw" else rng.uniform(0.02, 0.05) * w
    cx, cy = _pt(obj_box(im, cat), rng)
    d.ellipse([cx - r, cy - r * rng.uniform(0.6, 1.4), cx + r, cy + r * rng.uniform(0.6, 1.4)], fill=255)
    mask = mask.filter(ImageFilter.GaussianBlur(1))
    v = rng.randint(0, 60) if rng.random() < 0.6 else rng.randint(170, 255)
    col = (v, v, v) if cat == "screw" else tuple(min(255, max(0, v + rng.randint(-40, 40))) for _ in range(3))
    out.paste(Image.new("RGB", im.size, col), (0, 0), mask)
    return out


def syn_recolor(im, cat, rng):
    """Hue/brightness change of a large region (mimics wrong-colour wire / discolouration)."""
    w, h = im.size
    a = np.asarray(im).copy()
    hsv = cv2.cvtColor(a, cv2.COLOR_RGB2HSV).astype(np.int32)
    mask = np.zeros((h, w), np.uint8)
    cx, cy = _pt(obj_box(im, cat), rng)
    r = (0.03 if cat == "screw" else 0.12) * w
    cv2.circle(mask, (int(cx), int(cy)), int(r), 1, -1)
    hsv[..., 0] = np.where(mask > 0, (hsv[..., 0] + rng.randint(40, 140)) % 180, hsv[..., 0])
    hsv[..., 2] = np.where(mask > 0, np.clip(hsv[..., 2] * rng.uniform(0.6, 1.3), 0, 255), hsv[..., 2])
    out = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
    return Image.fromarray(out)


SYN = {"cutpaste": syn_cutpaste, "scar": syn_scar, "blob": syn_blob, "recolor": syn_recolor}


def f1_at(y, s, t):
    p = s > t
    tp = (p & (y == 1)).sum(); fp = (p & (y == 0)).sum(); fn = (~p & (y == 1)).sum()
    return 2 * tp / max(1, 2 * tp + fp + fn)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data"); ap.add_argument("out")
    ap.add_argument("--folds", type=int, default=3)
    ap.add_argument("--cats", default="bottle,cable,hazelnut,screw")
    ap.add_argument("--pct", type=float, default=99, help="threshold = this percentile of OOF good scores")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    tr = pd.read_csv(os.path.join(a.data, "train.csv"))
    te = pd.read_csv(os.path.join(a.data, "test.csv"))
    clips = os.path.join(a.data, "clips")
    model = Extractor()
    report = {}
    rows = []

    for cat in a.cats.split(","):
        if os.path.exists(os.path.join(a.out, f"test_{cat}.csv")):
            print(f"[{cat}] already done, skipping", flush=True)
            continue
        cfg, t0 = CFG[cat], time.time()
        pre = [preprocess(Image.open(os.path.join(clips, p)).convert("RGB"), cat, cfg["size"])
               for p in tr[tr.category == cat].path]
        imgs = [p[0] for p in pre]
        isg = np.array([p[1] for p in pre])
        gimgs = [prep.to_gray(im) for im in imgs]
        two = cat != "screw"     # screw is all-gray: single bank
        print(f"[{cat}] n={len(imgs)} gray_in_train={isg.sum()} prep {time.time() - t0:.0f}s", flush=True)

        rng = random.Random(0)
        idx = list(range(len(imgs))); rng.shuffle(idx)
        folds = [idx[i::a.folds] for i in range(a.folds)]
        oof = {"color": np.full(len(imgs), np.nan), "gray": np.full(len(imgs), np.nan)}
        syn = {b: {k: [] for k in SYN} for b in oof}
        for fi, hold in enumerate(folds):
            fit = [i for i in idx if i not in set(hold)]
            for b in (["color", "gray"] if two else ["gray"]):
                bank = Bank(model, [gimgs[i] for i in fit] if b == "gray" else [imgs[i] for i in fit if not isg[i]], cfg)
                hs = [i for i in hold if (b == "gray" or not isg[i])]
                src = [gimgs[i] for i in hs] if b == "gray" else [imgs[i] for i in hs]
                oof[b][hs] = bank.score(src)
                for name, fn in SYN.items():
                    if b == "gray" and name == "recolor":
                        continue
                    syn[b][name].extend(bank.score([fn(im, cat, rng) for im in src]))
                del bank
            print(f"[{cat}] fold {fi + 1}/{a.folds} ({time.time() - t0:.0f}s)", flush=True)

        res, thr = {}, {}
        for b in (["color", "gray"] if two else ["gray"]):
            g = oof[b][~np.isnan(oof[b])]
            allsyn = np.concatenate([np.array(v) for v in syn[b].values() if len(v)])
            y = np.r_[np.zeros(len(g)), np.ones(len(allsyn))]
            s = np.r_[g, allsyn]
            r = {"n_good": len(g), "auroc_all": round(float(roc_auc_score(y, s)), 4)}
            for name, v in syn[b].items():
                if len(v):
                    r["auroc_" + name] = round(float(roc_auc_score(np.r_[np.zeros(len(g)), np.ones(len(v))], np.r_[g, v])), 4)
            # proxy F1 with a 2:1 defect:good mix (defect-heavy test), sampled by weighting goods x2
            yw = np.r_[np.zeros(2 * len(g)), np.ones(len(allsyn))]
            sw = np.r_[g, g, allsyn]
            for p in PCTS:
                t = float(np.percentile(g, p))
                r[f"f1@p{p}"] = round(float(f1_at(yw, sw, t)), 4)
                r[f"recall@p{p}"] = round(float((allsyn > t).mean()), 4)
            thr[b] = float(np.percentile(g, a.pct))
            r["thr"] = thr[b]
            res[b] = r
            np.save(os.path.join(a.out, f"oof_{cat}_{b}.npy"), g)
        print(f"[{cat}] {json.dumps(res)}", flush=True)

        # final: banks from all train goods, score test
        tdf = te[te.category == cat]
        tpre = [preprocess(Image.open(os.path.join(clips, p)).convert("RGB"), cat, cfg["size"]) for p in tdf.path]
        tb = ["gray" if (g or not two) else "color" for _, g in tpre]
        crow = {}
        for b in (["color", "gray"] if two else ["gray"]):
            sel = [j for j in range(len(tpre)) if tb[j] == b]
            if not sel:
                continue
            bank = Bank(model, gimgs if b == "gray" else [im for im, g in zip(imgs, isg) if not g], cfg)
            for j, s in zip(sel, bank.score([tpre[j][0] for j in sel])):
                crow[j] = {"id": tdf.id.iloc[j], "category": cat, "bank": b, "score": float(s), "thr": thr[b]}
            del bank
        sc = pd.DataFrame([crow[j] for j in range(len(tpre))])
        sc.to_csv(os.path.join(a.out, f"test_{cat}.csv"), index=False)
        np.savez(os.path.join(a.out, f"cv_{cat}.npz"), **{f"{b}_{k}": np.array(v) for b in syn for k, v in syn[b].items()},
                 **{f"{b}_good": oof[b] for b in oof})
        json.dump(res, open(os.path.join(a.out, f"report_{cat}.json"), "w"), indent=1)
        res["test_pred_defect_frac"] = round(float((sc.score > sc.thr).mean()), 3)
        res["test_gray_frac"] = round(float((sc.bank == "gray").mean()), 3)
        json.dump(res, open(os.path.join(a.out, f"report_{cat}.json"), "w"), indent=1)
        print(f"[{cat}] test defect frac {res['test_pred_defect_frac']}  ({time.time() - t0:.0f}s)", flush=True)

    done = [c for c in CFG if os.path.exists(os.path.join(a.out, f"test_{c}.csv"))]
    pd.concat([pd.read_csv(os.path.join(a.out, f"test_{c}.csv")) for c in done]).to_csv(os.path.join(a.out, "test_scores.csv"), index=False)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
