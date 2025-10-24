#!/usr/bin/env python3
# /home/pi/ocr_sanity_probe.py
import os, re, glob, argparse, sys
import cv2, numpy as np

RE_CR = re.compile(r"CR\s*(1616|1620|2016|2025|2032|1650|1632)", re.I)
RE_DIG = re.compile(r"(1616|1620|2016|2025|2032|1650|1632)")

def norm_model(txt: str) -> str:
    if not txt: return ""
    t = txt.upper().replace(" ", "")
    m = RE_CR.search(t) or RE_DIG.search(t)
    return f"CR{m.group(1)}" if m else ""

def prep_variants(bgr, max_w=480):
    if bgr is None or bgr.size == 0:
        return []
    h,w = bgr.shape[:2]
    if w > max_w:
        s = max_w/float(w)
        bgr = cv2.resize(bgr, (int(w*s), int(h*s)), interpolation=cv2.INTER_LINEAR)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    # CLAHE + 锐化 + 轻去噪（与异步脚本思路一致）
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    g = clahe.apply(gray)
    blur = cv2.GaussianBlur(g, (3,3), 0)
    sharp = cv2.addWeighted(g, 1.2, blur, -0.2, 0)
    # 多种二值化 + 0°/180°翻转
    bins = []
    _, otsu = cv2.threshold(sharp, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    inv = cv2.bitwise_not(otsu)
    ada = cv2.adaptiveThreshold(sharp,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,31,7)
    for img in [sharp, otsu, inv, ada]:
        rgb0 = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        rgb180 = cv2.rotate(rgb0, cv2.ROTATE_180)
        bins.extend([rgb0, rgb180])
    return bins

def run_paddle(img, use_onnx=False):
    try:
        from paddleocr import PaddleOCR
        ocr = PaddleOCR(lang='en', use_angle_cls=False, det=False, rec=True, use_onnx=use_onnx)
    except Exception as e:
        return False, f"Paddle init fail: {e}", "", 0.0
    variants = prep_variants(img)
    best_raw, best_conf = "", 0.0
    for v in variants:
        try:
            res = ocr.ocr(v)  # 兼容多种返回格式
            # 解析（你的异步/fast文件也做了健壮解析）
            texts, probs = [], []
            if isinstance(res, list) and len(res)==1 and isinstance(res[0], list):
                res = res[0]
            if isinstance(res, list):
                for it in res:
                    if isinstance(it, dict):
                        txt = it.get('text') or it.get('transcription') or it.get('label') or ''
                        sc  = it.get('score') or it.get('confidence') or it.get('prob') or 0.0
                        if txt: texts.append(str(txt).strip()); probs.append(float(sc) if sc else 0.0)
                    elif isinstance(it, (list,tuple)) and len(it)>=2:
                        cand = it[1]
                        if isinstance(cand,(list,tuple)) and len(cand)>=2 and isinstance(cand[0],str):
                            texts.append(cand[0].strip()); probs.append(float(cand[1]) if len(cand)>1 else 0.0)
            raw = " ".join(texts).upper()
            conf = max(probs) if probs else 0.0
            if conf > best_conf:
                best_conf, best_raw = conf, raw
        except Exception as e:
            pass
    return True, best_raw, norm_model(best_raw), float(best_conf)

def run_tesseract(img):
    try:
        import pytesseract
    except Exception as e:
        return False, f"Tesseract import fail: {e}", "", 0.0
    variants = prep_variants(img)
    best_raw, best_score = "", -1e9
    best_avg = 0.0
    for v in variants:
        gray = cv2.cvtColor(v, cv2.COLOR_RGB2GRAY)
        cfg='--psm 6 -c tessedit_char_whitelist=C0123456789R.'
        try:
            d = pytesseract.image_to_data(gray, output_type=pytesseract.Output.DICT, config=cfg)
            txt = " ".join([w for w in d.get("text", []) if w]).upper()
            confs = [float(c) for c in d.get("conf", []) if str(c).isdigit() or (isinstance(c,(int,float)) and c>=0)]
            avg = sum(confs)/len(confs) if confs else 0.0
            # 打分：命中CRxxxx>纯数字>长度，类似你异步脚本里的策略
            score = (100.0 if RE_CR.search(txt) else (60.0 if RE_DIG.search(txt) else 0.0)) + min(len(txt),40)*0.5 + avg*0.1
            if score > best_score:
                best_score, best_raw, best_avg = score, txt, avg
        except Exception:
            pass
    return True, best_raw, norm_model(best_raw), float(best_avg)

def main():
    ap = argparse.ArgumentParser("OCR sanity probe (Paddle vs Tesseract)")
    ap.add_argument("--img", help="single image path")
    ap.add_argument("--dir", help="folder with images")
    ap.add_argument("--paddle-onnx", action="store_true", help="use PaddleOCR ONNX backend")
    ap.add_argument("--tess", action="store_true", help="also run Tesseract for comparison")
    args = ap.parse_args()

    paths = []
    if args.img and os.path.isfile(args.img):
        paths = [args.img]
    elif args.dir and os.path.isdir(args.dir):
        paths = sorted([p for ext in ("*.jpg","*.png","*.jpeg","*.bmp") for p in glob.glob(os.path.join(args.dir, ext))])
    else:
        print("Please provide --img or --dir"); sys.exit(1)

    print(f"Files: {len(paths)} | Paddle ONNX={args.paddle_onnx} | Tesseract={args.tess}")
    for p in paths:
        bgr = cv2.imread(p)
        okP, rawP, modelP, probP = run_paddle(bgr, use_onnx=args.paddle_onnx)
        line = f"[PADDLE] {os.path.basename(p)} | model={modelP or '?'} | prob={probP:.3f} | raw={rawP[:60]}"
        if not okP: line = f"[PADDLE] {os.path.basename(p)} | INIT_FAIL | {rawP}"
        print(line)
        if args.tess:
            okT, rawT, modelT, avgT = run_tesseract(bgr)
            line = f"[TESS ] {os.path.basename(p)} | model={modelT or '?'} | avg={avgT:.1f} | raw={rawT[:60]}"
            if not okT: line = f"[TESS ] {os.path.basename(p)} | INIT_FAIL | {rawT}"
            print(line)

if __name__ == "__main__":
    main()
