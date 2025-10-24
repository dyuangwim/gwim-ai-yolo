#!/usr/bin/env python3
# /home/pi/ocr_try_lite.py
# Goal: 针对 test.jpg（或任意图）进行“强预处理 + 轻量OCR”，优先 RapidOCR(ONNX, PPOCRv4-mobile)，
#       回退 PaddleOCR（若安装成功），再回退 Tesseract。输出原文方便你校对。

import os, sys, argparse
import cv2
import numpy as np

def center_circle_mask(img, ratio=0.90):
    h, w = img.shape[:2]
    r = int(min(h, w) * ratio / 2)
    mask = np.zeros((h, w), np.uint8)
    cv2.circle(mask, (w//2, h//2), r, 255, -1)
    return mask

def suppress_glare(gray):
    # 双边 + 顶帽抑制高光
    g = cv2.bilateralFilter(gray, d=7, sigmaColor=50, sigmaSpace=7)
    se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9,9))
    top = cv2.morphologyEx(g, cv2.MORPH_TOPHAT, se)
    return cv2.subtract(g, cv2.convertScaleAbs(top, alpha=0.9))

def enhance(gray):
    # CLAHE + 锐化
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    g = clahe.apply(gray)
    blur = cv2.GaussianBlur(g, (3,3), 0)
    return cv2.addWeighted(g, 1.35, blur, -0.35, 0)

def make_variants(bgr):
    # 居中裁成正方形并放大到 1200
    h, w = bgr.shape[:2]
    s = min(h, w)
    x0, y0 = (w - s)//2, (h - s)//2
    sq = bgr[y0:y0+s, x0:x0+s].copy()
    if s < 1200:
        scale = 1200.0 / s
        sq = cv2.resize(sq, (int(s*scale), int(s*scale)), interpolation=cv2.INTER_CUBIC)

    gray = cv2.cvtColor(sq, cv2.COLOR_BGR2GRAY)
    # 圆形遮罩（屏蔽外圈塑料/凹坑）
    mask = center_circle_mask(gray, ratio=0.92)
    gray = cv2.bitwise_and(gray, mask)

    # 抑制高光 + 增强
    sup   = suppress_glare(gray)
    sharp = enhance(sup)

    # 多种阈值化
    _, otsu = cv2.threshold(sharp, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    inv  = cv2.bitwise_not(otsu)
    ada  = cv2.adaptiveThreshold(sharp,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                 cv2.THRESH_BINARY, 41, 5)

    # 轻开运算去噪
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (2,2))
    otsu = cv2.morphologyEx(otsu, cv2.MORPH_OPEN, k, iterations=1)
    ada  = cv2.morphologyEx(ada,  cv2.MORPH_OPEN, k, iterations=1)

    outs = []
    for g in [sharp, otsu, inv, ada]:
        rgb0 = cv2.cvtColor(g, cv2.COLOR_GRAY2RGB)
        rgb1 = cv2.rotate(rgb0, cv2.ROTATE_180)
        outs.extend([rgb0, rgb1])

    # 额外强对比一版
    strong = cv2.convertScaleAbs(sharp, alpha=1.6, beta=-15)
    outs.append(cv2.cvtColor(strong, cv2.COLOR_GRAY2RGB))
    return outs  # [RGB图...]

def try_rapidocr(imgs):
    try:
        from rapidocr_onnxruntime import RapidOCR
    except Exception as e:
        return ["[RapidOCR] NOT_AVAILABLE: " + str(e)]
    ocr = RapidOCR()  # 第一次会自动下模型
    outs = []
    for i, im in enumerate(imgs):
        try:
            res, _ = ocr(im)  # 返回 [ [ (box, text, score), ... ] ]
            texts = []
            for item in (res or []):
                if isinstance(item, (list, tuple)) and len(item) >= 2:
                    t = item[1]
                    if t: texts.append(t)
            txt = " ".join(texts).strip()
            if txt:
                outs.append(f"[RapidOCR][v{i:02d}] {txt}")
        except Exception:
            pass
    return outs or ["[RapidOCR] (no text)"]

def try_paddle(imgs):
    try:
        from paddleocr import PaddleOCR
        ocr = PaddleOCR(lang='en')  # 简单初始化，避免参数兼容问题
    except Exception as e:
        return ["[PaddleOCR] NOT_AVAILABLE: " + str(e)]
    outs = []
    for i, im in enumerate(imgs):
        try:
            res = ocr.ocr(im)
            # 统一解析
            if isinstance(res, list) and len(res)==1 and isinstance(res[0], list):
                res = res[0]
            lines = []
            if isinstance(res, list):
                for it in res:
                    if isinstance(it, dict):
                        t = it.get('text') or it.get('transcription') or it.get('label') or ''
                        if t: lines.append(t)
                    elif isinstance(it, (list, tuple)) and len(it)>=2:
                        cand = it[1]
                        if isinstance(cand, (list,tuple)) and len(cand)>=1 and isinstance(cand[0], str):
                            lines.append(cand[0])
            txt = " ".join(lines).strip()
            if txt:
                outs.append(f"[PaddleOCR][v{i:02d}] {txt}")
        except Exception:
            pass
    return outs or ["[PaddleOCR] (no text)"]

def try_tesseract(imgs):
    try:
        import pytesseract
    except Exception as e:
        return ["[Tesseract] NOT_AVAILABLE: " + str(e)]
    psms = [6, 7, 11, 13]
    outs = []
    for i, im in enumerate(imgs):
        gray = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)
        for p in psms:
            try:
                txt = pytesseract.image_to_string(gray, config=f"--oem 3 --psm {p}")
                txt = txt.replace("\n", " ").strip()
                if txt:
                    outs.append(f"[Tesseract][v{i:02d} psm{p}] {txt}")
            except Exception:
                pass
    return outs or ["[Tesseract] (no text)"]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--img", default="test.jpg", help="image path")
    ap.add_argument("--save-debug", action="store_true", help="save variants to ./_lite_debug")
    args = ap.parse_args()

    if not os.path.isfile(args.img):
        print("Image not found:", args.img); sys.exit(1)
    bgr = cv2.imread(args.img)
    if bgr is None:
        print("Failed to read image"); sys.exit(1)

    variants = make_variants(bgr)
    if args.save_debug:
        os.makedirs("_lite_debug", exist_ok=True)
        for i, im in enumerate(variants):
            cv2.imwrite(os.path.join("_lite_debug", f"v{i:02d}.png"), cv2.cvtColor(im, cv2.COLOR_RGB2BGR))
        print(f"Saved {len(variants)} variants to ./_lite_debug")

    print(">>> RapidOCR (Paddle Lite models via ONNX)")
    for line in try_rapidocr(variants):
        print(line)

    print("\n>>> PaddleOCR (if installed)")
    for line in try_paddle(variants):
        print(line)

    print("\n>>> Tesseract (fallback)")
    for line in try_tesseract(variants):
        print(line)

if __name__ == "__main__":
    main()
