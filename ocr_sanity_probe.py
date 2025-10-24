#!/usr/bin/env python3
# /home/pi/ocr_one_shot_strong.py
# 目的：对 test.jpg 进行强力预处理 + OCR，尽量把任意可读文本提取出来（不限定 CRxxxx）。

import os, sys, argparse, math
import cv2
import numpy as np

def center_circle_mask(img, ratio=0.62):
    h, w = img.shape[:2]
    r = int(min(h, w) * ratio / 2) * 2
    cx, cy = w // 2, h // 2
    mask = np.zeros((h, w), np.uint8)
    cv2.circle(mask, (cx, cy), r, 255, -1)
    return mask

def suppress_glare(gray):
    # 抑制高光：顶帽 & 限幅
    # 1) 平滑
    g = cv2.bilateralFilter(gray, d=7, sigmaColor=50, sigmaSpace=7)
    # 2) 顶帽突出反光，再减去一部分
    se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9,9))
    top = cv2.morphologyEx(g, cv2.MORPH_TOPHAT, se)
    sup = cv2.subtract(g, cv2.convertScaleAbs(top, alpha=0.7))
    return sup

def enhance(gray):
    # CLAHE + 锐化
    clahe = cv2.createCLAHE(clipLimit=2.4, tileGridSize=(8,8))
    g = clahe.apply(gray)
    blur = cv2.GaussianBlur(g, (3,3), 0)
    sharp = cv2.addWeighted(g, 1.35, blur, -0.35, 0)
    return sharp

def variants_from(bgr):
    # 1) 中心裁方形 + 圆形遮罩（屏蔽外圈凹坑与塑料边）
    h, w = bgr.shape[:2]
    s = min(h, w)
    x0, y0 = (w - s)//2, (h - s)//2
    sq = bgr[y0:y0+s, x0:x0+s].copy()

    # 2) 放大到宽 1200（Tesseract 很吃分辨率）
    if s < 1200:
        scale = 1200.0 / s
        sq = cv2.resize(sq, (int(s*scale), int(s*scale)), interpolation=cv2.INTER_CUBIC)

    gray = cv2.cvtColor(sq, cv2.COLOR_BGR2GRAY)

    # 3) 圆形遮罩
    mask = center_circle_mask(gray, ratio=0.88)
    gray = cv2.bitwise_and(gray, mask)

    # 4) 抑制高光 + 增强
    sup = suppress_glare(gray)
    sharp = enhance(sup)

    # 5) 多种二值化
    _, otsu = cv2.threshold(sharp, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    inv  = cv2.bitwise_not(otsu)
    ada  = cv2.adaptiveThreshold(sharp,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                 cv2.THRESH_BINARY, 31, 7)
    # 轻形态学（去椒盐）
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (2,2))
    otsu = cv2.morphologyEx(otsu, cv2.MORPH_OPEN, k, iterations=1)
    ada  = cv2.morphologyEx(ada,  cv2.MORPH_OPEN, k, iterations=1)

    # 6) 生成 RGB 及 180° 旋转版本
    outs = []
    for g in [sharp, otsu, inv, ada]:
        rgb0 = cv2.cvtColor(g, cv2.COLOR_GRAY2RGB)
        rgb1 = cv2.rotate(rgb0, cv2.ROTATE_180)
        outs.extend([rgb0, rgb1])

    # 7) 额外提供一个“强对比”版本
    strong = cv2.convertScaleAbs(sharp, alpha=1.6, beta=-20)
    strong = cv2.cvtColor(strong, cv2.COLOR_GRAY2RGB)
    outs.append(strong)
    return outs

def run_tesseract(imgs):
    try:
        import pytesseract
    except Exception as e:
        return ["[TESS] INIT_FAIL: "+str(e)]

    # 多种 PSM 尝试：6=单块文字；7=单行；11=稀疏文本；13=原始；都测一遍
    psms = [6, 7, 11, 13]
    results = []
    for idx, im in enumerate(imgs):
        gray = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)
        for psm in psms:
            cfg = f'--oem 1 --psm {psm}'
            try:
                txt = pytesseract.image_to_string(gray, config=cfg)
                txt = txt.replace('\n', ' ').strip()
                if txt:
                    results.append(f"[TESS][v{idx:02d} psm{psm}] {txt}")
            except Exception:
                pass
    if not results:
        results = ["[TESS] (no text)"]
    return results

def run_paddle(imgs):
    # Paddle 在你机子上参数经常不兼容，这里只做“能用就用”
    try:
        from paddleocr import PaddleOCR
        ocr = PaddleOCR(lang='en')  # 不传 det/rec/use_angle_cls 等参数
    except Exception as e:
        return [f"[PADDLE] INIT_FAIL: {e}"]

    outs = []
    for idx, im in enumerate(imgs):
        try:
            res = ocr.ocr(im)
            # 解析结果
            lines = []
            if isinstance(res, list) and len(res)==1 and isinstance(res[0], list):
                res = res[0]
            if isinstance(res, list):
                for it in res:
                    if isinstance(it, dict):
                        t = it.get('text') or it.get('transcription') or it.get('label') or ''
                        if t: lines.append(t)
                    elif isinstance(it, (list,tuple)) and len(it)>=2:
                        cand = it[1]
                        if isinstance(cand,(list,tuple)) and len(cand)>=1 and isinstance(cand[0],str):
                            lines.append(cand[0])
            txt = " ".join(lines).strip()
            if txt:
                outs.append(f"[PADDLE][v{idx:02d}] {txt}")
        except Exception:
            pass
    if not outs:
        outs = ["[PADDLE] (no text)"]
    return outs

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--img", default="test.jpg", help="path to image")
    ap.add_argument("--save-debug", action="store_true", help="save preprocessed variants to ./_debug_ocr")
    args = ap.parse_args()

    if not os.path.isfile(args.img):
        print("Image not found:", args.img); sys.exit(1)
    bgr = cv2.imread(args.img)
    if bgr is None:
        print("Failed to read image"); sys.exit(1)

    vars_ = variants_from(bgr)
    if args.save_debug:
        os.makedirs("_debug_ocr", exist_ok=True)
        for i, im in enumerate(vars_):
            cv2.imwrite(os.path.join("_debug_ocr", f"v{i:02d}.png"), cv2.cvtColor(im, cv2.COLOR_RGB2BGR))
        print(f"saved {_debug_count(len(vars_))} variants into ./_debug_ocr")

    print(">>> Tesseract results")
    for line in run_tesseract(vars_):
        print(line)

    print("\n>>> Paddle results")
    for line in run_paddle(vars_):
        print(line)

def _debug_count(n): return n

if __name__ == "__main__":
    main()
