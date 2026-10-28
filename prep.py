"""Preprocessing: denoise, grayscale detection, screw pose canonicalization."""
import cv2
import numpy as np
from PIL import Image


def denoise(im, h=7):
    a = np.asarray(im.convert("RGB"))
    a = cv2.fastNlMeansDenoisingColored(a, None, h, h, 5, 15)
    return Image.fromarray(a)


def is_gray(im):
    a = np.asarray(im.convert("RGB")).astype(np.float32)
    a = cv2.GaussianBlur(a, (5, 5), 0)
    return float(np.abs(a[..., 0] - a[..., 1]).mean() + np.abs(a[..., 1] - a[..., 2]).mean()) < 2.0


def to_gray(im):
    return Image.fromarray(np.asarray(im.convert("L"))).convert("RGB")


def screw_align(im, out=(448, 160), margin=0.08):
    """Rotate screw so its long axis is horizontal with the head on the left, crop to it."""
    g = np.asarray(im.convert("L")).astype(np.uint8)
    H, W = g.shape
    b = cv2.GaussianBlur(g, (5, 5), 0)
    _, m = cv2.threshold(b, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m)
    if isinstance(out, int):
        out = (out, out)
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
        ax = -ax
        proj = -proj
    ang = np.degrees(np.arctan2(ax[1], ax[0]))
    L = proj.max() - proj.min()
    mid = c + ax * (proj.max() + proj.min()) / 2
    M = cv2.getRotationMatrix2D((float(mid[0]), float(mid[1])), ang, 1.0)
    side = L * (1 + 2 * margin)
    s = OW / side
    M[0, 2] += side / 2 - mid[0]
    M[1, 2] += side * OH / OW / 2 - mid[1]
    M *= s
    a = np.asarray(im.convert("RGB"))
    r = cv2.warpAffine(a, M, (OW, OH), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    return Image.fromarray(r)


# per-category noise cut-offs on noise_sigma(): clean images sit well below, noisy ones well above
NOISE_THR = {"bottle": 3.0, "cable": 4.5, "hazelnut": 3.0, "screw": 3.0}


def noise_sigma(im):
    """Immerkaer noise estimate (std of additive noise) on the luminance channel."""
    a = np.asarray(im.convert("L")).astype(np.float32)
    k = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], np.float32)
    return float(np.sqrt(np.pi / 2) * np.abs(cv2.filter2D(a, -1, k)).mean() / 6)
