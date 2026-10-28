"""PatchCore v3: leave-one-out calibrated thresholds + multi-backbone ensemble.

Changes vs v2:
  - One memory bank per (category, colour/gray bank, backbone) built from ALL train goods. Every train good is
    scored against that same bank with its own patches (and its rotated copies) masked out, so good-score
    statistics match how test images are scored. v2 used 2/3-size CV banks, which inflated good scores and
    pushed thresholds too high (lost recall on the leaderboard).
  - Synthetic defects are scored the same leave-one-out way (proxy for model selection only).
  - Backbones: wide_resnet50_2 (layer2+3) and densenet201 (denseblock2+3), both ImageNet-pretrained.
Writes <out>/<cat>_<bank>_<backbone>.npz with test / good / syn scores; ensemble.py combines them.
"""
import argparse, math, os, random, time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torchvision
from PIL import Image

import prep
from pc2 import CFG, MEAN, STD, SYN, EMB, PROJ, preprocess, rot_aug, to_tensor
import cv2
from scipy.ndimage import gaussian_filter

torch.set_num_threads(12)


class WRN50(torch.nn.Module):
    def __init__(self):
        super().__init__()
        m = torchvision.models.wide_resnet50_2(weights="IMAGENET1K_V1").eval()
        self.stem = torch.nn.Sequential(m.conv1, m.bn1, m.relu, m.maxpool, m.layer1)
        self.l2, self.l3 = m.layer2, m.layer3

    def feats(self, x):
        f2 = self.l2(self.stem(x))
        return f2, self.l3(f2)


class WRN101(torch.nn.Module):
    def __init__(self):
        super().__init__()
        m = torchvision.models.wide_resnet101_2(weights="IMAGENET1K_V1").eval()
        self.stem = torch.nn.Sequential(m.conv1, m.bn1, m.relu, m.maxpool, m.layer1)
        self.l2, self.l3 = m.layer2, m.layer3

    def feats(self, x):
        f2 = self.l2(self.stem(x))
        return f2, self.l3(f2)


class Dense201(torch.nn.Module):
    def __init__(self):
        super().__init__()
        f = torchvision.models.densenet201(weights="IMAGENET1K_V1").features.eval()
        self.a = torch.nn.Sequential(f.conv0, f.norm0, f.relu0, f.pool0, f.denseblock1, f.transition1, f.denseblock2)
        self.b = torch.nn.Sequential(f.transition2, f.denseblock3)

    def feats(self, x):
        f2 = self.a(x)
        return f2, self.b(f2)


class Extractor(torch.nn.Module):
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


BACKBONES = {"wrn50": WRN50, "wrn101": WRN101, "dn201": Dense201}


def embed(model, imgs, bs=8):
    return torch.cat([model(to_tensor(imgs[i:i + bs])).half() for i in range(0, len(imgs), bs)])


def coreset_idx(z16, k):
    """Greedy k-center on a random projection; returns selected row indices."""
    n = z16.shape[0]
    k = min(k, n)
    g = torch.Generator().manual_seed(0)
    proj = torch.randn(z16.shape[1], PROJ, generator=g) / math.sqrt(PROJ)
    z = torch.empty(n, PROJ)
    for i in range(0, n, 65536):
        z[i:i + 65536] = z16[i:i + 65536].float() @ proj
    zn = (z * z).sum(1)
    sel = [0]
    mind = zn - 2 * (z @ z[0]) + zn[0]
    for _ in range(k - 1):
        j = int(torch.argmax(mind))
        sel.append(j)
        torch.minimum(mind, zn - 2 * torch.mv(z, z[j]) + zn[j], out=mind)
    return torch.tensor(sel)


class Bank:
    """Memory bank that remembers which train image every stored patch came from."""

    def __init__(self, model, imgs, cfg, sids=None):
        """sids: train-image id of every entry in imgs (default 0..n-1); used for leave-one-out masking."""
        self.model, self.cfg = model, cfg
        sids = np.arange(len(imgs)) if sids is None else np.asarray(sids)
        src = rot_aug(imgs) if cfg["rot"] else imgs
        # rot_aug returns the originals followed by 3 rotations of each image in turn
        sid = np.concatenate([sids, np.repeat(sids, 3)]) if cfg["rot"] else sids
        if cfg["pos"]:
            feats = embed(model, src)
            N, H, W, D = feats.shape
            self.hw = (H, W)
            self.pos = feats.float().reshape(N, H * W, D).permute(1, 0, 2).contiguous()  # HW,N,D
            self.pos_n = (self.pos ** 2).sum(-1)                                         # HW,N
            self.sid = torch.from_numpy(sid)
            return
        K, CH = cfg["coreset"], 32
        nch = math.ceil(len(src) / CH)
        keep, keep_sid = [], []
        for c in range(nch):
            f = embed(model, src[c * CH:(c + 1) * CH])
            self.hw = f.shape[1:3]
            P = f.shape[1] * f.shape[2]
            f = f.reshape(-1, EMB)
            s = np.repeat(sid[c * CH:(c + 1) * CH], P)
            idx = coreset_idx(f, max(1, min(len(f), 2 * K // nch + 1)))
            keep.append(f[idx]); keep_sid.append(s[idx.numpy()])
            del f
        allf, allsid = torch.cat(keep), np.concatenate(keep_sid)
        idx = coreset_idx(allf, K)
        self.mem = allf[idx].float()
        self.sid = torch.from_numpy(allsid[idx.numpy()])

    def patch_dist(self, f, ex=None):
        H, W = self.hw
        if not self.cfg["pos"]:
            d = torch.cdist(f.reshape(-1, EMB).float(), self.mem)
            if ex is not None:
                d[:, self.sid == ex] = float("inf")
            return d.min(1).values.reshape(H, W)
        pos_n = self.pos_n
        if ex is not None:
            pos_n = pos_n.clone(); pos_n[:, self.sid == ex] = float("inf")
        r = self.cfg.get("radius", 2)
        t = f.float()
        best = torch.full((H, W), float("inf"))
        for dy in range(-r, r + 1):
            for dx in range(-r, r + 1):
                ts = torch.zeros(H, W, EMB)
                ts[max(0, dy):H + min(0, dy), max(0, dx):W + min(0, dx)] = t[max(0, -dy):H - max(0, dy), max(0, -dx):W - max(0, dx)]
                tv = ts.reshape(H * W, EMB)
                cross = torch.bmm(self.pos, tv.unsqueeze(-1)).squeeze(-1)
                d2 = (pos_n - 2 * cross).min(1).values + (tv ** 2).sum(-1)
                d = d2.clamp_min(0).sqrt().reshape(H, W)
                best[max(0, -dy):H - max(0, dy), max(0, -dx):W - max(0, dx)] = torch.minimum(
                    best[max(0, -dy):H - max(0, dy), max(0, -dx):W - max(0, dx)],
                    d[max(0, dy):H + min(0, dy), max(0, dx):W + min(0, dx)])
        return best

    def score(self, imgs, ex=None, bs=8):
        """ex: per-image train index whose patches are masked out (leave-one-out), or None."""
        out = []
        for i in range(0, len(imgs), bs):
            feats = self.model(to_tensor(imgs[i:i + bs])).half()
            for j, f in enumerate(feats):
                d = self.patch_dist(f, None if ex is None else int(ex[i + j])).numpy()
                W, H = imgs[0].size
                amap = cv2.resize(d, (W, H), interpolation=cv2.INTER_LINEAR)
                out.append(float(gaussian_filter(amap, sigma=4).max()))
        return np.array(out)


def prep_img(raw, cat, size, cond, force_dn=False):
    """v4 preprocessing. cond=True: NLM-denoise only images whose noise estimate is above the category cut-off
    (v3 denoised everything, which also smooths away fine scratches). Returns (image, is_gray, was_denoised)."""
    dn = force_dn or not cond or prep.noise_sigma(raw) > prep.NOISE_THR[cat]
    im = prep.denoise(raw) if dn else raw
    if cat == "screw":
        return prep.screw_align(prep.to_gray(im), size), True, dn
    g = prep.is_gray(im)
    if g:
        im = prep.to_gray(im)
    return im.resize(size, Image.BILINEAR), g, dn


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data"); ap.add_argument("out")
    ap.add_argument("--cats", default="bottle,cable,hazelnut,screw")
    ap.add_argument("--backbones", default="wrn50,dn201")
    ap.add_argument("--size", default=None, help="override input size W,H (e.g. 640,224)")
    ap.add_argument("--suffix", default="", help="appended to output npz names, e.g. _hr")
    ap.add_argument("--cond-denoise", action="store_true", help="denoise only noisy images")
    ap.add_argument("--bank-dn", action="store_true", help="with --cond-denoise: also add denoised copies of clean train images to the bank")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    tr = pd.read_csv(os.path.join(a.data, "train.csv"))
    te = pd.read_csv(os.path.join(a.data, "test.csv"))
    clips = os.path.join(a.data, "clips")

    for cat in a.cats.split(","):
        cfg, two = dict(CFG[cat]), cat != "screw"
        if a.size:
            cfg["size"] = tuple(int(v) for v in a.size.split(","))
        banks = ["color", "gray"] if two else ["gray"]
        todo = [(b, m) for m in a.backbones.split(",") for b in banks
                if not os.path.exists(os.path.join(a.out, f"{cat}_{b}_{m}{a.suffix}.npz"))]
        if not todo:
            print(f"[{cat}] done, skipping", flush=True); continue
        t0 = time.time()
        raws = [Image.open(os.path.join(clips, p)).convert("RGB") for p in tr[tr.category == cat].path]
        pre = [prep_img(r, cat, cfg["size"], a.cond_denoise) for r in raws]
        imgs, isg, wasdn = [p[0] for p in pre], np.array([p[1] for p in pre]), np.array([p[2] for p in pre])
        gimgs = [prep.to_gray(im) for im in imgs]
        # optional extra bank members: denoised copies of the clean train images (same source id)
        xi = np.where(~wasdn)[0] if a.bank_dn else np.array([], int)
        ximgs = [prep_img(raws[i], cat, cfg["size"], True, force_dn=True)[0] for i in xi]
        del raws
        tdf = te[te.category == cat].reset_index(drop=True)
        tpre = [prep_img(Image.open(os.path.join(clips, p)).convert("RGB"), cat, cfg["size"], a.cond_denoise) for p in tdf.path]
        print(f"[{cat}] denoised train {wasdn.sum()}/{len(wasdn)} test {sum(p[2] for p in tpre)}/{len(tpre)} extra bank {len(xi)}", flush=True)
        tbank = np.array(["gray" if (g or not two) else "color" for _, g, _ in tpre])
        print(f"[{cat}] prep {time.time() - t0:.0f}s", flush=True)

        for m in a.backbones.split(","):
            if not any(mm == m for _, mm in todo):
                continue
            model = Extractor(BACKBONES[m]()).eval()
            for b in banks:
                f = os.path.join(a.out, f"{cat}_{b}_{m}{a.suffix}.npz")
                if os.path.exists(f):
                    continue
                # bank members: gray bank = gray copies of all goods; colour bank = colour goods only
                mem_idx = np.arange(len(imgs)) if b == "gray" else np.where(~isg)[0]
                src = [gimgs[i] if b == "gray" else imgs[i] for i in mem_idx]
                pos = {int(i): k for k, i in enumerate(mem_idx)}
                xk = [k for k, i in enumerate(xi) if int(i) in pos]
                xsrc = [prep.to_gray(ximgs[k]) if b == "gray" else ximgs[k] for k in xk]
                bank = Bank(model, src + xsrc, cfg, sids=np.r_[np.arange(len(src)), [pos[int(xi[k])] for k in xk]].astype(int))
                sel = np.where(tbank == b)[0]
                s_test = bank.score([tpre[j][0] for j in sel])
                s_good = bank.score(src, ex=np.arange(len(src)))
                rng = random.Random(1)
                names = [k for k in SYN if not (b == "gray" and k == "recolor")]
                kinds = [names[i % len(names)] for i in range(len(src))]
                syn = [SYN[k](im, cat, rng) for k, im in zip(kinds, src)]
                s_syn = bank.score(syn, ex=np.arange(len(src)))
                np.savez(f, test_ids=tdf.id.values[sel], test=s_test, good=s_good, syn=s_syn, syn_kind=np.array(kinds))
                del bank
                print(f"[{cat}] {b}/{m} n_bank={len(src)}+{len(xsrc)} n_test={len(sel)} ({time.time() - t0:.0f}s)", flush=True)
            del model
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
