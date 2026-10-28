"""DefectVision - complete pipeline in one file (Kaggle notebook or script).

Unsupervised defect detection with PatchCore on frozen ImageNet backbones. Train = good images only;
test = good + defective; metric F1 with defective as positive. This reproduces our best submission (v5):

  preprocessing   NLM denoise every image; detect grayscale photos; screws rotated horizontal (head left) and cropped
  memory banks    per (category, colour/gray, backbone, size); gray bank = gray copies of all train goods;
                  bottle/hazelnut banks include 90/180/270 rotations; cable uses position-aware nearest neighbour
  calibration     every train good is scored against the full bank with its own patches masked (leave-one-out),
                  giving the good-score distribution used for z-scores and thresholds
  ensemble        bottle, hazelnut: wrn50 + dn201 @256 | cable: dn201 @224 + @288 | screw: dn201 @640x224 + @800x280
  thresholds      gap in the sorted test z-scores (fallback: p97.5 of LOO goods); overrides cable_gray=1.4, screw_gray=1.0

Kaggle: add the competition data, and either turn Internet ON (torchvision downloads the ImageNet weights) or attach
a dataset holding wide_resnet50_2-95faca4d.pth and densenet201-c1103571.pth. GPU optional but ~10x faster. Writes /kaggle/working/submission.csv. Per-run scores are cached in scores/, so a re-run resumes.
Locally: python kaggle_defectvision.py <data_dir> [out_dir]
"""
import glob, math, os, random, sys, time
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torchvision
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter
from scipy.ndimage import gaussian_filter
from sklearn.metrics import roc_auc_score

# ============================== settings ==============================
CATS = ["bottle", "cable", "hazelnut", "screw"]
RUNS = [  # (category, backbone, (W, H)) - every run scores both banks of its category
    ("bottle", "wrn50", (256, 256)), ("bottle", "dn201", (256, 256)),
    ("hazelnut", "wrn50", (256, 256)), ("hazelnut", "dn201", (256, 256)),
    ("cable", "dn201", (224, 224)), ("cable", "dn201", (288, 288)),
    ("screw", "dn201", (640, 224)), ("screw", "dn201", (800, 280)),
]
CFG = {  # rot: add rotated copies to the bank; pos: position-aware matching within +-radius feature cells
    "bottle":   dict(rot=True,  pos=False),
    "cable":    dict(rot=False, pos=True, radius=2),
    "hazelnut": dict(rot=True,  pos=False),
    "screw":    dict(rot=False, pos=False),
}
CORESET = 12000           # patches kept per memory bank
EMB, PROJ = 512, 128      # patch feature size; random-projection size used by the coreset search
PCT, MIN_GAP = 97.5, 0.5  # fallback threshold percentile of LOO good z-scores; minimum width of a usable gap
OVERRIDE = {"cable_gray": 1.4, "screw_gray": 1.0}  # fixed z thresholds (chosen on the public leaderboard)
SYN_CHECK = True          # also score synthetic defects -> proxy AUROC per bank (diagnostic only, ~1.5x run time)

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cuda.matmul.allow_tf32 = False  # keep distances in full fp32 precision
torch.backends.cudnn.allow_tf32 = False
if DEV.type == "cpu":
    torch.set_num_threads(os.cpu_count() or 4)
MEAN = torch.tensor([0.485, 0.456, 0.406], device=DEV).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225], device=DEV).view(1, 3, 1, 1)
BILINEAR = Image.Resampling.BILINEAR
ROTS = (Image.Transpose.ROTATE_90, Image.Transpose.ROTATE_180, Image.Transpose.ROTATE_270)


# ============================== preprocessing ==============================
def denoise(im, h=7):
    return Image.fromarray(cv2.fastNlMeansDenoisingColored(np.asarray(im.convert("RGB")), None, h, h, 5, 15))


def is_gray(im):
    a = cv2.GaussianBlur(np.asarray(im.convert("RGB")).astype(np.float32), (5, 5), 0)
    return float(np.abs(a[..., 0] - a[..., 1]).mean() + np.abs(a[..., 1] - a[..., 2]).mean()) < 2.0


def to_gray(im):
    return Image.fromarray(np.asarray(im.convert("L"))).convert("RGB")


def screw_align(im, out, margin=0.08):
    """Rotate screw so its long axis is horizontal with the head on the left, crop to it."""
    g = np.asarray(im.convert("L")).astype(np.uint8)
    b = cv2.GaussianBlur(g, (5, 5), 0)
    _, m = cv2.threshold(b, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m)
    OW, OH = out
    if n < 2:
        return im.resize(out)
    k = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    ys, xs = np.nonzero(lab == k)
    pts = np.stack([xs, ys], 1).astype(np.float64)
    c = pts.mean(0)
    _, _, vt = np.linalg.svd(pts - c, full_matrices=False)
    ax = vt[0]
    proj = (pts - c) @ ax
    perp = np.abs((pts - c) @ vt[1])
    # head = end with larger perpendicular extent
    lo, hi = proj < np.percentile(proj, 15), proj > np.percentile(proj, 85)
    if perp[hi].mean() > perp[lo].mean():
        ax, proj = -ax, -proj
    ang = np.degrees(np.arctan2(ax[1], ax[0]))
    L = proj.max() - proj.min()
    mid = c + ax * (proj.max() + proj.min()) / 2
    M = cv2.getRotationMatrix2D((float(mid[0]), float(mid[1])), ang, 1.0)
    side = L * (1 + 2 * margin)
    M[0, 2] += side / 2 - mid[0]
    M[1, 2] += side * OH / OW / 2 - mid[1]
    M *= OW / side
    r = cv2.warpAffine(np.asarray(im.convert("RGB")), M, (OW, OH), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    return Image.fromarray(r)


def finish(dn, cat, size):
    """Denoised image -> (model input, is_gray)."""
    if cat == "screw":
        return screw_align(to_gray(dn), size), True
    g = is_gray(dn)
    return (to_gray(dn) if g else dn).resize(size, BILINEAR), g


_DN = {}


def load_denoised(img_dir, paths):
    """NLM denoising is size-independent and the slowest preprocessing step: do it once per image, in parallel."""
    todo = [p for p in paths if p not in _DN]
    def work(p):
        return p, denoise(Image.open(os.path.join(img_dir, p)).convert("RGB"))
    with ThreadPoolExecutor(os.cpu_count() or 4) as ex:
        _DN.update(ex.map(work, todo))
    return [_DN[p] for p in paths]


# ============================== backbones ==============================
class WRN50(torch.nn.Module):
    def __init__(self):
        super().__init__()
        m = torchvision.models.wide_resnet50_2(weights="IMAGENET1K_V1")
        self.stem = torch.nn.Sequential(m.conv1, m.bn1, m.relu, m.maxpool, m.layer1)
        self.l2, self.l3 = m.layer2, m.layer3

    def feats(self, x):
        f2 = self.l2(self.stem(x))
        return f2, self.l3(f2)


class Dense201(torch.nn.Module):
    def __init__(self):
        super().__init__()
        f = torchvision.models.densenet201(weights="IMAGENET1K_V1").features
        self.a = torch.nn.Sequential(f.conv0, f.norm0, f.relu0, f.pool0, f.denseblock1, f.transition1, f.denseblock2)
        self.b = torch.nn.Sequential(f.transition2, f.denseblock3)

    def feats(self, x):
        f2 = self.a(x)
        return f2, self.b(f2)


BACKBONES = {"wrn50": WRN50, "dn201": Dense201}


class Extractor(torch.nn.Module):
    """Mid-level patch features: layer2 + upsampled layer3, 3x3 local averaging, reduced to EMB dims."""

    def __init__(self, net):
        super().__init__()
        self.net = net

    @torch.no_grad()
    def forward(self, x):
        f2, f3 = self.net.feats((x - MEAN) / STD)
        f2 = F.avg_pool2d(f2, 3, 1, 1)
        f3 = F.avg_pool2d(f3, 3, 1, 1)
        f3 = F.interpolate(f3, size=f2.shape[-2:], mode="bilinear", align_corners=False)
        f = torch.cat([f2, f3], 1)
        B, C, H, W = f.shape
        f = f.permute(0, 2, 3, 1).reshape(-1, 1, C)
        return F.adaptive_avg_pool1d(f, EMB).reshape(B, H, W, EMB)


def to_tensor(imgs):
    arr = np.stack([np.asarray(im, dtype=np.float32) / 255.0 for im in imgs])
    return torch.from_numpy(arr).permute(0, 3, 1, 2).to(DEV)


def embed(model, imgs, bs=8):
    return torch.cat([model(to_tensor(imgs[i:i + bs])).half() for i in range(0, len(imgs), bs)])  # N,H,W,D fp16


# ============================== PatchCore memory bank ==============================
def coreset_idx(z16, k):
    """Greedy k-center selection on a random projection; returns selected row indices."""
    n = z16.shape[0]
    k = min(k, n)
    g = torch.Generator().manual_seed(0)
    proj = (torch.randn(z16.shape[1], PROJ, generator=g) / math.sqrt(PROJ)).to(DEV)
    z = torch.empty(n, PROJ, device=DEV)
    for i in range(0, n, 65536):
        z[i:i + 65536] = z16[i:i + 65536].float() @ proj
    zn = (z * z).sum(1)
    sel = [0]
    mind = zn - 2 * (z @ z[0]) + zn[0]
    for _ in range(k - 1):
        j = int(torch.argmax(mind))
        sel.append(j)
        torch.minimum(mind, zn - 2 * torch.mv(z, z[j]) + zn[j], out=mind)
    return torch.tensor(sel, device=DEV)


class Bank:
    """Memory bank of good patches that remembers which train image every stored patch came from."""

    def __init__(self, model, imgs, cfg):
        self.model, self.cfg = model, cfg
        sids = np.arange(len(imgs))
        # rotated copies follow the originals (3 per image, in order) and keep their source id
        src = imgs + [im.transpose(t) for im in imgs for t in ROTS] if cfg["rot"] else imgs
        sid = np.concatenate([sids, np.repeat(sids, 3)]) if cfg["rot"] else sids
        if cfg["pos"]:  # keep every patch, indexed by grid position
            feats = embed(model, src)
            N, H, W, D = feats.shape
            self.hw = (H, W)
            self.pos = feats.float().reshape(N, H * W, D).permute(1, 0, 2).contiguous()  # HW,N,D
            self.pos_n = (self.pos ** 2).sum(-1)                                         # HW,N
            self.sid = torch.from_numpy(sid).to(DEV)
            return
        # low-memory coreset: subsample each chunk of images, then take the coreset of the union
        CH = 32
        nch = math.ceil(len(src) / CH)
        keep, keep_sid = [], []
        for c in range(nch):
            f = embed(model, src[c * CH:(c + 1) * CH])
            self.hw = f.shape[1:3]
            P = f.shape[1] * f.shape[2]
            f = f.reshape(-1, EMB)
            s = np.repeat(sid[c * CH:(c + 1) * CH], P)
            idx = coreset_idx(f, max(1, min(len(f), 2 * CORESET // nch + 1)))
            keep.append(f[idx]); keep_sid.append(s[idx.cpu().numpy()])
            del f
        allf, allsid = torch.cat(keep), np.concatenate(keep_sid)
        idx = coreset_idx(allf, CORESET)
        self.mem = allf[idx].float()
        self.sid = torch.from_numpy(allsid[idx.cpu().numpy()]).to(DEV)

    def patch_dist(self, f, ex=None):
        """f: H,W,D -> H,W nearest-neighbour distance; ex: train-image id whose patches are masked out."""
        H, W = self.hw
        if not self.cfg["pos"]:
            d = torch.cdist(f.reshape(-1, EMB).float(), self.mem)
            if ex is not None:
                d[:, self.sid == ex] = float("inf")
            return d.min(1).values.reshape(H, W)
        pos_n = self.pos_n
        if ex is not None:
            pos_n = pos_n.clone(); pos_n[:, self.sid == ex] = float("inf")
        r = self.cfg["radius"]
        t = f.float()
        best = torch.full((H, W), float("inf"), device=DEV)
        for dy in range(-r, r + 1):
            for dx in range(-r, r + 1):
                # compare test[h, w] with bank[:, h+dy, w+dx]: shift the test map instead of the bank
                ts = torch.zeros(H, W, EMB, device=DEV)
                ts[max(0, dy):H + min(0, dy), max(0, dx):W + min(0, dx)] = t[max(0, -dy):H - max(0, dy), max(0, -dx):W - max(0, dx)]
                tv = ts.reshape(H * W, EMB)
                cross = torch.bmm(self.pos, tv.unsqueeze(-1)).squeeze(-1)  # HW,N
                d2 = (pos_n - 2 * cross).min(1).values + (tv ** 2).sum(-1)
                d = d2.clamp_min(0).sqrt().reshape(H, W)
                # d is indexed at bank position (h+dy, w+dx); move it back to test position (h, w)
                best[max(0, -dy):H - max(0, dy), max(0, -dx):W - max(0, dx)] = torch.minimum(
                    best[max(0, -dy):H - max(0, dy), max(0, -dx):W - max(0, dx)],
                    d[max(0, dy):H + min(0, dy), max(0, dx):W + min(0, dx)])
        return best

    def score(self, imgs, ex=None, bs=8):
        """Image score = max of the smoothed, upsampled anomaly map. ex: per-image train id to mask (LOO)."""
        out = []
        for i in range(0, len(imgs), bs):
            feats = self.model(to_tensor(imgs[i:i + bs])).half()
            for j, f in enumerate(feats):
                d = self.patch_dist(f, None if ex is None else int(ex[i + j])).cpu().numpy()
                W, H = imgs[0].size
                amap = cv2.resize(d, (W, H), interpolation=cv2.INTER_LINEAR)
                out.append(float(gaussian_filter(amap, sigma=4).max()))
        return np.array(out)


# ============================== synthetic defects (validation proxy only) ==============================
def obj_box(im, cat):
    w, h = im.size
    if cat == "screw":  # aligned: screw along the horizontal mid line
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
    hsv = cv2.cvtColor(np.asarray(im).copy(), cv2.COLOR_RGB2HSV).astype(np.int32)
    mask = np.zeros((h, w), np.uint8)
    cx, cy = _pt(obj_box(im, cat), rng)
    cv2.circle(mask, (int(cx), int(cy)), int((0.03 if cat == "screw" else 0.12) * w), 1, -1)
    hsv[..., 0] = np.where(mask > 0, (hsv[..., 0] + rng.randint(40, 140)) % 180, hsv[..., 0])
    hsv[..., 2] = np.where(mask > 0, np.clip(hsv[..., 2] * rng.uniform(0.6, 1.3), 0, 255), hsv[..., 2])
    return Image.fromarray(cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB))


SYN = {"cutpaste": syn_cutpaste, "scar": syn_scar, "blob": syn_blob, "recolor": syn_recolor}


# ============================== scoring runs ==============================
def run_name(cat, bank, m, size):
    return f"{cat}_{bank}_{m}_{size[0]}x{size[1]}"


def score_run(cat, m, size, tr, te, img_dir, score_dir):
    """Build both banks for one (category, backbone, size) and save test / LOO-good / synthetic scores."""
    cfg, two = CFG[cat], cat != "screw"  # screw images are all gray: one bank
    banks = ["color", "gray"] if two else ["gray"]
    todo = [b for b in banks if not os.path.exists(os.path.join(score_dir, run_name(cat, b, m, size) + ".npz"))]
    if not todo:
        print(f"[{cat} {m} {size}] cached", flush=True); return
    t0 = time.time()
    pre = [finish(d, cat, size) for d in load_denoised(img_dir, list(tr[tr.category == cat].path))]
    imgs, isg = [p[0] for p in pre], np.array([p[1] for p in pre])
    gimgs = [to_gray(im) for im in imgs]
    tdf = te[te.category == cat].reset_index(drop=True)
    tpre = [finish(d, cat, size) for d in load_denoised(img_dir, list(tdf.path))]
    tbank = np.array(["gray" if (g or not two) else "color" for _, g in tpre])
    model = Extractor(BACKBONES[m]()).to(DEV).eval()
    for b in todo:
        # gray bank = gray copies of all goods (gray test photos); colour bank = colour goods only
        src = gimgs if b == "gray" else [im for im, g in zip(imgs, isg) if not g]
        bank = Bank(model, src, cfg)
        sel = np.where(tbank == b)[0]
        res = dict(test_ids=tdf.id.values[sel], test=bank.score([tpre[j][0] for j in sel]),
                   good=bank.score(src, ex=np.arange(len(src))))
        if SYN_CHECK:
            rng = random.Random(1)
            names = [k for k in SYN if not (b == "gray" and k == "recolor")]
            syn = [SYN[names[i % len(names)]](im, cat, rng) for i, im in enumerate(src)]
            res["syn"] = bank.score(syn, ex=np.arange(len(src)))
        np.savez(os.path.join(score_dir, run_name(cat, b, m, size) + ".npz"), **res)
        del bank
        if DEV.type == "cuda":
            torch.cuda.empty_cache()
        print(f"[{cat} {m} {size}] {b}: bank={len(src)} test={len(sel)} ({time.time() - t0:.0f}s)", flush=True)
    del model


# ============================== ensemble + thresholds ==============================
def make_submission(te, score_dir):
    rows = []
    for cat in CATS:
        for b in ["color", "gray"]:
            fs = [os.path.join(score_dir, run_name(c, b, m, s) + ".npz") for c, m, s in RUNS if c == cat]
            if not any(os.path.exists(f) for f in fs):
                continue
            d = [np.load(f, allow_pickle=True) for f in fs]
            ids = d[0]["test_ids"]
            keys = ["test", "good"] + (["syn"] if all("syn" in x.files for x in d) else [])
            # standardise each model by its own LOO good scores, then average
            z = []
            for x in d:
                assert (x["test_ids"] == ids).all()
                mu, sd = x["good"].mean(), x["good"].std() + 1e-9
                z.append({k: (x[k] - mu) / sd for k in keys})
            e = {k: np.mean([zz[k] for zz in z], 0) for k in keys}
            thr, how = np.percentile(e["good"], PCT), f"p{PCT:g}"
            # low-density separation: widest empty stretch of sorted test scores between good p90 and max+1
            lo, hi = np.percentile(e["good"], 90), e["good"].max() + 1.0
            s = np.sort(e["test"]); m = (s[:-1] >= lo) & (s[1:] <= hi)
            if m.any():
                gaps = np.diff(s)[m]; k = int(np.argmax(gaps))
                if gaps[k] >= MIN_GAP:
                    thr, how = (s[:-1][m][k] + s[1:][m][k]) / 2, f"gap {s[:-1][m][k]:.2f}-{s[1:][m][k]:.2f}"
            if f"{cat}_{b}" in OVERRIDE:
                thr, how = OVERRIDE[f"{cat}_{b}"], "override"
            lab = (e["test"] > thr).astype(int)
            auc = (f"synAUROC={roc_auc_score(np.r_[0 * e['good'], 1 + 0 * e['syn']], np.r_[e['good'], e['syn']]):.4f} "
                   if "syn" in e else "")
            print(f"{cat:9s}{b:6s} n={len(ids):3d} {auc}thr={thr:.2f} ({how}) flag={lab.mean():.2f} "
                  f"goodFPR={(e['good'] > thr).mean():.3f}")
            rows.append(pd.DataFrame({"id": ids, "category": cat, "bank": b, "score": e["test"], "thr": thr, "label": lab}))
    sc = pd.concat(rows)
    sub = te[["id"]].merge(sc, on="id", how="left")
    if sub.label.isna().any():
        raise SystemExit(f"{sub.label.isna().sum()} test images have no score; no submission written")
    sub["label"] = sub.label.astype(int)
    return sub


def find_data():
    hits = glob.glob("/kaggle/input/**/train.csv", recursive=True)
    if not hits:
        raise SystemExit("train.csv not found under /kaggle/input - add the competition data to the notebook")
    return os.path.dirname(hits[0])


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("-")]  # notebooks pass their own -f flag
    on_kaggle = os.path.exists("/kaggle/input")
    data = args[0] if args else (find_data() if on_kaggle else ".")
    out = args[1] if len(args) > 1 else ("/kaggle/working" if on_kaggle else data)
    score_dir = os.path.join(out, "scores")
    os.makedirs(score_dir, exist_ok=True)
    tr, te = pd.read_csv(os.path.join(data, "train.csv")), pd.read_csv(os.path.join(data, "test.csv"))
    first = tr.path.iloc[0]
    img_dir = next((os.path.dirname(p) for p in glob.glob(os.path.join(data, "**", first), recursive=True)), None)
    if img_dir is None:
        raise SystemExit(f"image {first} not found under {data}")
    print(f"device={DEV} data={data} images={img_dir} train={len(tr)} test={len(te)}", flush=True)

    t0 = time.time()
    for cat, m, size in RUNS:
        score_run(cat, m, size, tr, te, img_dir, score_dir)
        if all(c != cat for c, _, _ in RUNS[RUNS.index((cat, m, size)) + 1:]):
            for p in list(tr[tr.category == cat].path) + list(te[te.category == cat].path):
                _DN.pop(p, None)  # category finished: free its denoised images
    print(f"all runs done ({time.time() - t0:.0f}s)\n", flush=True)

    sub = make_submission(te, score_dir)
    sub.to_csv(os.path.join(score_dir, "test_scores.csv"), index=False)
    sub[["id", "label"]].to_csv(os.path.join(out, "submission.csv"), index=False)
    print(f"\ndefective frac {sub.label.mean():.3f} {sub.groupby('category').label.mean().round(3).to_dict()}")
    print("wrote", os.path.join(out, "submission.csv"))


if __name__ == "__main__":
    main()
