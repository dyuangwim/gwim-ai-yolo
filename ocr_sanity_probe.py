#!/usr/bin/env python3
# /home/pi/ocr_one_shot.py
# 单张图 OCR (PaddleOCR 可选 + Tesseract)，仅用于验证“OCR 本身能不能读出 CRxxxx”。

import os, sys, re, argparse
import cv2
import numpy as np

RE_CR = re.compile(r"CR\s*(1616|1620|2016|2025|2032|1632|1650)", re.I)
RE_DIG = re.compile(r"(1616|1620|2016|2025|2032|1632|1650)")

def norm_model(text: str) -> str:
    if not text: return ""
    t = text.upper().replace(" ", "")
    m = RE_CR.search(t) or RE_DIG.search(t)
    return f"CR{m.group(1)}" if m else ""

def prep_variants(bgr):
    """尽量简单但有效的几种增强：缩放->灰度->CLAHE->轻锐化->多阈值->(0°/180°)。"""
    if bgr is None or bgr.size == 0:
        return []
    # 放大到宽不小于 640（Tesseract 对像素很敏感）
    h, w = bgr.shape[:2]
    if w < 640:
        s = 640.0 / w
        bgr = cv2.resize(bgr, (int(w*s), int(h*s)), interpolation=cv2.INTER_CUBIC)

    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.2, tileGridSize=(8,8))
    g = clahe.apply(gray)
    blur = cv2.GaussianBlur(g, (3,3), 0)
    sharp = cv2.addWeighted(g, 1.25, blur, -0.25, 0)

    # 试三种二值化
    _, otsu = cv2.threshold(sharp, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    inv  = cv2.bitwise_not(otsu)
    ada  = cv2.adaptiveThreshold(sharp,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                 cv2.THRESH_BINARY, 31, 7)

    # 轻形态学，压噪点
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (2,2))
    otsu = cv2.morphologyEx(otsu, cv2.MORPH_OPEN, k, iterations=1)
    ada  = cv2.morphologyEx(ada,  cv2.MORPH_OPEN, k, iterations=1)

    # 0°/180° 两个角度都跑（电池常倒置）
    def to_rgb(x): return cv2.cvtColor(x, cv2.COLOR_GRAY2RGB)
    vars_ = []
    for img in [sharp, otsu, inv, ada]:
        rgb0 = to_rgb(img)
        rgb180 = cv2.rotate(rgb0, cv2.ROTATE_180)
        vars_.extend([rgb0, rgb180])
    return vars_

def ocr_with_paddle(img_rgb):
    """若 PaddleOCR 可用则读取一遍；否则返回不可用状态。"""
    try:
        from paddleocr import PaddleOCR
        # 不传 use_onnx —— 你的环境报过 Unknown argument: use_onnx
        ocr = PaddleOCR(lang='en', use_angle_cls=False, det=False, rec=True)
    except Exception as e:
        return False, f"[PADDLE] INIT_FAIL: {e}", "", 0.0

    variants = prep_variants(cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))
    best_raw, best_conf = "", 0.0
    for v in variants:
        try:
            res = ocr.ocr(v)
            # 兼容多种返回结构
            if isinstance(res, list) and len(res) == 1 and isinstance(res[0], list):
                res = res[0]
            texts, probs = [], []
            if isinstance(res, list):
                for it in res:
                    if isinstance(it, dict):
                        t = it.get('text') or it.get('transcription') or it.get('label') or ''
                        c = it.get('score') or it.get('confidence') or it.get('prob') or 0.0
                        if t: texts.append(str(t).strip()); probs.append(float(c) if c else 0.0)
                    elif isinstance(it, (list, tuple)) and len(it) >= 2:
                        cand = it[1]
                        if isinstance(cand, (list, tuple)) and len(cand) >= 2 and isinstance(cand[0], str):
                            texts.append(cand[0].strip()); probs.append(float(cand[1]) if len(cand)>1 else 0.0)
            raw = " ".join(texts).upper()
            conf = max(probs) if probs else 0.0
            if conf > best_conf:
                best_conf, best_raw = conf, raw
        except Exception:
            pass
    return True, best_raw, norm_model(best_raw), float(best_conf)

def ocr_with_tesseract(img_rgb):
    try:
        import pytesseract
    except Exception as e:
        return False, f"[TESS] INIT_FAIL: {e}", "", 0.0

    variants = prep_variants(cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))
    best_raw, best_score = "", -1e9
    for v in variants:
        gray = cv2.cvtColor(v, cv2.COLOR_RGB2GRAY)
        # psm 6：假设是一行/少量文本；白名单：只允许 CR 和数字与点
        cfg = '--psm 6 -c tessedit_char_whitelist=CR0123456789.'
        try:
            data = pytesseract.image_to_data(gray, output_type=pytesseract.Output.DICT, config=cfg)
            txt = " ".join([w for w in data.get("text", []) if w]).upper()
            # 简单打分：命中CR优先 + 长度微奖分
            score = (100 if RE_CR.search(txt) else (60 if RE_DIG.search(txt) else 0)) + min(len(txt), 40) * 0.5
            if score > best_score:
                best_score, best_raw = score, txt
        except Exception:
            pass
    return True, best_raw, norm_model(best_raw), 0.0

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--img", default="test.jpg", help="path to image (default: ./test.jpg)")
    args = ap.parse_args()

    if not os.path.isfile(args.img):
        print(f"Image not found: {args.img}")
        sys.exit(1)

    bgr = cv2.imread(args.img)
    if bgr is None:
        print("Failed to read image"); sys.exit(1)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    okP, rawP, modelP, probP = ocr_with_paddle(rgb)
    if okP:
        print(f"[PADDLE] model={modelP or '?'} | prob={probP:.3f} | raw={rawP}")
    else:
        print(rawP)  # INIT_FAIL reason

    okT, rawT, modelT, _ = ocr_with_tesseract(rgb)
    if okT:
        print(f"[TESS ] model={modelT or '?'} | raw={rawT}")
    else:
        print(rawT)

if __name__ == "__main__":
    main()
