#!/usr/bin/env python3
# /home/pi/ocr_lite_clean.py
# RapidOCR-ONNX（PaddleOCR Lite 模型）+ 针对金属电池的预处理
# 依赖：pip install rapidocr-onnxruntime opencv-python numpy

import os, sys, re, argparse
import cv2
import numpy as np

RE_CR = re.compile(r"\bCR\s*(1616|1620|2016|2025|2032|1632|1650)\b", re.I)
RE_EN = re.compile(r"\bENERGIZER\b", re.I)

def center_circle_mask(img, ratio=0.92):
    h, w = img.shape[:2]
    r = int(min(h, w) * ratio / 2)
    mask = np.zeros((h, w), np.uint8)
    cv2.circle(mask, (w//2, h//2), r, 255, -1)
    return mask

def suppress_glare(gray):
    g = cv2.bilateralFilter(gray, d=7, sigmaColor=50, sigmaSpace=7)
    se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9,9))
    top = cv2.morphologyEx(g, cv2.MORPH_TOPHAT, se)
    return cv2.subtract(g, cv2.convertScaleAbs(top, alpha=0.9))

def enhance(gray):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    g = clahe.apply(gray)
    blur = cv2.GaussianBlur(g, (3,3), 0)
    return cv2.addWeighted(g, 1.35, blur, -0.35, 0)

def make_variants(bgr):
    h, w = bgr.shape[:2]
    s = min(h, w)
    x0, y0 = (w - s)//2, (h - s)//2
    sq = bgr[y0:y0+s, x0:x0+s].copy()
    if s < 1200:
        scale = 1200.0 / s
        sq = cv2.resize(sq, (int(s*scale), int(s*scale)), interpolation=cv2.INTER_CUBIC)

    gray = cv2.cvtColor(sq, cv2.COLOR_BGR2GRAY)
    gray = cv2.bitwise_and(gray, center_circle_mask(gray, ratio=0.92))

    sup   = suppress_glare(gray)
    sharp = enhance(sup)

    _, otsu = cv2.threshold(sharp, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    inv    = cv2.bitwise_not(otsu)
    ada    = cv2.adaptiveThreshold(sharp,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, 41, 5)

    k = cv2.getStructuringElement(cv2.MORPH_RECT, (2,2))
    otsu = cv2.morphologyEx(otsu, cv2.MORPH_OPEN, k, iterations=1)
    ada  = cv2.morphologyEx(ada,  cv2.MORPH_OPEN, k, iterations=1)

    outs = []
    for g in [sharp, otsu, inv, ada]:
        rgb0 = cv2.cvtColor(g, cv2.COLOR_GRAY2RGB)
        rgb1 = cv2.rotate(rgb0, cv2.ROTATE_180)
        outs.extend([rgb0, rgb1])

    strong = cv2.convertScaleAbs(sharp, alpha=1.6, beta=-15)
    outs.append(cv2.cvtColor(strong, cv2.COLOR_GRAY2RGB))
    return outs

def normalize_text(t: str) -> str:
    s = t.upper()
    s = s.replace("CR2O25", "CR2025").replace("CR2O32", "CR2032")
    s = s.replace("ENERGIZERS", "ENERGIZER").replace("ENERGIZER®", "ENERGIZER")
    return s

def score_text(txt: str, conf: float) -> float:
    txtU = txt.upper()
    score = 0.0
    if RE_CR.search(txtU): score += 4.0
    if RE_EN.search(txtU): score += 2.0
    L = len(txtU)
    if 6 <= L <= 40: score += 1.0
    score += min(max(conf, 0.0), 1.0) * 2.0
    return score

def run_rapidocr_variants(imgs):
    from rapidocr_onnxruntime import RapidOCR
    ocr = RapidOCR()
    best = ("", 0.0, -1)
    lines = []
    for i, im in enumerate(imgs):
        try:
            res, _ = ocr(im)
            if not res: continue
            texts, confs = [], []
            for item in res:
                if isinstance(item, (list, tuple)) and len(item) >= 3:
                    t = str(item[1]).strip()
                    c = float(item[2]) if item[2] is not None else 0.0
                    if t:
                        texts.append(t); confs.append(c)
            if not texts: continue
            raw = normalize_text(" ".join(texts))
            conf = max(confs) if confs else 0.0
            sc = score_text(raw, conf)
            lines.append((i, raw, conf, sc))
            if sc > best[1]:
                best = (raw, sc, i)
        except Exception:
            continue
    return best, lines

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--img", default="test.jpg", help="image path")
    ap.add_argument("--save-debug", action="store_true",
                    help="save preprocessed variants to ./_lite_debug_best")
    args = ap.parse_args()

    if not os.path.isfile(args.img):
        print("Image not found:", args.img); sys.exit(1)
    bgr = cv2.imread(args.img)
    if bgr is None:
        print("Failed to read image"); sys.exit(1)

    variants = make_variants(bgr)
    best, lines = run_rapidocr_variants(variants)

    if args.save_debug:  # ← 修正点
        os.makedirs("_lite_debug_best", exist_ok=True)
        for i, im in enumerate(variants):
            cv2.imwrite(os.path.join("_lite_debug_best", f"v{i:02d}.png"),
                        cv2.cvtColor(im, cv2.COLOR_RGB2BGR))

    print("Top candidates (idx, conf, score, text):")
    for i, raw, conf, sc in sorted(lines, key=lambda x: x[3], reverse=True)[:6]:
        print(f"[v{i:02d}] conf={conf:.3f} score={sc:.2f} :: {raw}")

    if best[2] >= 0:
        print("\nBEST:")
        print(f"[v{best[2]:02d}] {best[0]}")
    else:
        print("No text found by RapidOCR")

if __name__ == "__main__":
    main()
