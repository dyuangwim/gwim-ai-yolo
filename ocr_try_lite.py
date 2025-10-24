#!/usr/bin/env python3
# /home/pi/ocr_cr_only.py
# 只抽取电池型号（CRxxxx）。优先 RapidOCR (PPOCR Lite / ONNX)，无则回退 Tesseract。
# 安装：
#   pip install rapidocr-onnxruntime opencv-python numpy
#   (可选) sudo apt-get install -y tesseract-ocr && pip install pytesseract

import os, sys, re, argparse
import cv2, numpy as np

# ---- 配置：允许的型号集合（你要增减就在这里改） ----
ALLOWED = {"CR1616", "CR1620", "CR1632", "CR1650", "CR2016", "CR2025", "CR2032"}

# ---- 预处理（和你前面跑通的一致，专打金属反光） ----
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
    gray = cv2.bitwise_and(gray, center_circle_mask(gray, 0.92))
    sup   = suppress_glare(gray)
    sharp = enhance(sup)

    _, otsu = cv2.threshold(sharp, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    inv    = cv2.bitwise_not(otsu)
    ada    = cv2.adaptiveThreshold(sharp, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
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

# ---- 文本清洗/纠错与提取 ----
RE_CR_RAW = re.compile(r"([CDGOQ]?R)\s*([0O]\s*\d\s*\d\s*\d|\d\s*\d\s*\d\s*\d)", re.I)  # 容忍 C/DR/GR、O→0 乱序
def normalize_text(t: str) -> str:
    s = t.upper()
    # 前缀纠正：DR/GR/QR/OR → CR
    s = re.sub(r"\b[DGQO]R", "CR", s)
    # 常见字符纠错
    s = (s.replace("O", "0").replace("S", "5").replace("Z", "2")
            .replace("B", "8").replace("G", "6").replace("I", "1").replace("L", "1"))
    # 统一多余空格
    s = re.sub(r"\s+", " ", s).strip()
    return s

def extract_models(txt: str):
    txt = normalize_text(txt)
    cand = set()
    # 先抓紧邻的 CR + 4位数字
    for m in re.finditer(r"CR\s*(\d{4})", txt):
        cand.add("CR"+m.group(1))
    # 再用更宽松的匹配兜底
    if not cand:
        for m in RE_CR_RAW.finditer(txt):
            digits = re.sub(r"\s+", "", m.group(2))
            if len(digits) == 4 and digits.isdigit():
                cand.add("CR"+digits)
    # 过滤到允许列表（如果你要允许更多，在 ALLOWED 里加）
    cand_allowed = [c for c in cand if c in ALLOWED]
    return cand_allowed or list(cand)  # 若都不在白名单，先返回原候选供你观察

# ---- OCR 引擎（优先 RapidOCR，失败则 Tesseract） ----
def run_rapidocr_variants(imgs):
    try:
        from rapidocr_onnxruntime import RapidOCR
    except Exception:
        return []  # 不可用
    ocr = RapidOCR()
    outs = []
    for i, im in enumerate(imgs):
        try:
            res, _ = ocr(im)  # [(box, text, score), ...]
            if not res: continue
            # 以最大 score 的行为主参考，同时把整张聚合文本也存一份
            best_conf = 0.0; all_text = []
            for item in res:
                if len(item) >= 3:
                    all_text.append(str(item[1]))
                    sc = float(item[2]) if item[2] is not None else 0.0
                    best_conf = max(best_conf, sc)
            outs.append((" ".join(all_text), best_conf, f"v{i:02d}"))
        except Exception:
            pass
    return outs

def run_tesseract_variants(imgs):
    try:
        import pytesseract
    except Exception:
        return []
    outs = []
    for i, im in enumerate(imgs):
        gray = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)
        for psm in (6, 7, 11, 13):
            try:
                txt = pytesseract.image_to_string(gray, config=f"--oem 3 --psm {psm}")
                txt = txt.replace("\n", " ").strip()
                if txt:
                    outs.append((txt, 0.0, f"v{i:02d}-psm{psm}"))  # Tesseract无统一置信度，这里置 0
            except Exception:
                pass
    return outs

def score_models(models, conf):
    # 打分：命中白名单多 → 高分；有一个也行。再加 OCR 置信度权重。
    mscore = 0
    if not models:
        return conf * 0.5
    for m in models:
        if m in ALLOWED: mscore += 1.0
    return mscore * 2.0 + conf

def main():
    ap = argparse.ArgumentParser("Extract CRxxxx only")
    ap.add_argument("--img", help="single image path")
    ap.add_argument("--dir", help="dir of images")
    ap.add_argument("--save-debug", action="store_true", help="save variants")
    args = ap.parse_args()

    paths = []
    if args.img and os.path.isfile(args.img):
        paths = [args.img]
    elif args.dir and os.path.isdir(args.dir):
        exts = (".jpg",".jpeg",".png",".bmp")
        paths = [os.path.join(args.dir, f) for f in sorted(os.listdir(args.dir)) if f.lower().endswith(exts)]
    else:
        print("Please provide --img or --dir"); sys.exit(1)

    for p in paths:
        bgr = cv2.imread(p)
        if bgr is None:
            print(f"{p} -> read fail"); continue
        variants = make_variants(bgr)

        # 保存预处理图，方便你肉眼挑
        if args.save_debug:
            outdir = os.path.join(os.path.dirname(p) or ".", "_cr_debug")
            os.makedirs(outdir, exist_ok=True)
            for i, im in enumerate(variants):
                cv2.imwrite(os.path.join(outdir, f"{os.path.basename(p)}_v{i:02d}.png"),
                            cv2.cvtColor(im, cv2.COLOR_RGB2BGR))

        # OCR
        results = run_rapidocr_variants(variants)
        if not results:
            results = run_tesseract_variants(variants)

        # 聚合 & 选择最佳
        best = ("", [], -1, -1.0)  # (raw, models, idx, score)
        for idx, (txt, conf, tag) in enumerate(results):
            models = extract_models(txt)
            sc = score_models(models, conf)
            if sc > best[3]:
                best = (txt, models, idx, sc)

        raw, models, _, sc = best
        # 输出：只给型号（按出现顺序去重）
        uniq = []
        for m in models:
            if m not in uniq:
                uniq.append(m)
        model_str = ",".join(uniq) if uniq else "?"
        print(f"{os.path.basename(p)} -> {model_str}   (score={sc:.2f})")
        # 若需要，也可打印原文：# print("RAW:", raw)

if __name__ == "__main__":
    main()
