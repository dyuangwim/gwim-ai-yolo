#!/usr/bin/env python3
# /home/pi/ocr_lite_clean.py
# 使用 RapidOCR-ONNX（PaddleOCR Lite 模型）+ 针对金属电池的预处理，
# 自动挑选最佳候选并做文本清洗，输出干净的结果。
# 依赖：pip install rapidocr-onnxruntime opencv-python numpy

import os, sys, re, argparse
import cv2
import numpy as np

RE_CR = re.compile(r"\bCR\s*([12][06]16|2016|2025|2032|1632|1650)\b", re.I)  # 支持常见型号
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
    # 居中裁正方形 + 放大到宽≥1200（小字对OCR很关键）
    h, w = bgr.shape[:2]
    s = min(h, w)
    x0, y0 = (w - s)//2, (h - s)//2
    sq = bgr[y0:y0+s, x0:x0+s].copy()
    if s < 1200:
        scale = 1200.0 / s
        sq = cv2.resize(sq, (int(s*scale), int(s*scale)), interpolation=cv2.INTER_CUBIC)

    gray = cv2.cvtColor(sq, cv2.COLOR_BGR2GRAY)
    gray = cv2.bitwise_and(gray, center_circle_mask(gray, ratio=0.92))  # 只读电池面

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
    # 修正常见误读：O/0, S/5, Z/2, G/6, I/1 等，仅对短串做最小替换
    s = t.upper()
    # 先修CR型号串附近的易错
    s = re.sub(r"CR20Z5", "CR2025", s)
    s = re.sub(r"CR20S5", "CR2025", s)
    s = re.sub(r"CR203S", "CR2035", s)  # 以防万一
    s = s.replace("CR2O25", "CR2025").replace("CR2O32","CR2032")  # O→0
    s = s.replace("CR20S5", "CR2025").replace("CR203S","CR2035")
    # 品牌常见花样
    s = s.replace("ENERGIZERS", "ENERGIZER").replace("ENERGIZER®", "ENERGIZER")
    return s

def score_text(txt: str, conf: float) -> float:
    # 打分：包含CR型号加分、包含ENERGIZER加分、长度适中加分、引擎置信度加权
    txtU = txt.upper()
    score = 0.0
    if RE_CR.search(txtU): score += 4.0
    if RE_EN.search(txtU): score += 2.0
    L = len(txtU)
    if 6 <= L <= 40: score += 1.0
    score += min(max(conf, 0.0), 1.0) * 2.0  # 置信度（0~1）*2
    return score

def run_rapidocr_variants(imgs):
    from rapidocr_onnxruntime import RapidOCR
    ocr = RapidOCR()
    best = ("", 0.0, -1)  # (text, score, idx)
    all_lines = []

    for i, im in enumerate(imgs):
        try:
            res, _ = ocr(im)  # -> [ [(box, text, score), ...] ]
            if not res: continue
            # 聚合该 variant 的文字
            texts = []
            confs = []
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
            all_lines.append((i, raw, conf, sc))
            if sc > best[1]:
                best = (raw, sc, i)
        except Exception:
            continue
    return best, all_lines

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--img", default="test.jpg", help="image path")
    ap.add_argument("--save-debug", action="store_true", help="save preprocessed variants to ./_lite_debug_best")
    args = ap.parse_args()

    if not os.path.isfile(args.img):
        print("Image not found:", args.img); sys.exit(1)
    bgr = cv2.imread(args.img)
    if bgr is None:
        print("Failed to read image"); sys.exit(1)

    variants = make_variants(bgr)
    best, lines = run_rapidocr_variants(variants)  # (text, score, idx)

    if args.save-debug or args.save_debug:
        os.makedirs("_lite_debug_best", exist_ok=True)
        for i, im in enumerate(variants):
            cv2.imwrite(os.path.join("_lite_debug_best", f"v{i:02d}.png"),
                        cv2.cvtColor(im, cv2.COLOR_RGB2BGR))

    # 打印候选Top（按分数降序）
    lines_sorted = sorted(lines, key=lambda x: x[3], reverse=True)[:6]
    print("Top candidates (idx, conf, score, text):")
    for i, raw, conf, sc in lines_sorted:
        print(f"[v{i:02d}] conf={conf:.3f} score={sc:.2f} :: {raw}")

    if best[2] >= 0:
        print("\nBEST:")
        print(f"[v{best[2]:02d}] {best[0]}")
    else:
        print("No text found by RapidOCR")

if __name__ == "__main__":
    main()
