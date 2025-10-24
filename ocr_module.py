# ocr_module.py
import re, cv2, numpy as np

# 允许的型号（你可按需增减）
ALLOWED = {"CR1616","CR1650","CR2016","CR2025","CR2032"}

# —— 预处理（和你调通的逻辑一致：放大、遮罩、去高光、CLAHE、阈值、180°）——
def _center_circle_mask(img, ratio=0.92):
    h,w = img.shape[:2]
    r = int(min(h,w)*ratio/2)
    mask = np.zeros((h,w), np.uint8)
    cv2.circle(mask, (w//2, h//2), r, 255, -1)
    return mask

def _suppress_glare(gray):
    g = cv2.bilateralFilter(gray, d=7, sigmaColor=50, sigmaSpace=7)
    se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9,9))
    top = cv2.morphologyEx(g, cv2.MORPH_TOPHAT, se)
    return cv2.subtract(g, cv2.convertScaleAbs(top, alpha=0.9))

def _enhance(gray):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    g = clahe.apply(gray)
    blur = cv2.GaussianBlur(g, (3,3), 0)
    return cv2.addWeighted(g, 1.35, blur, -0.35, 0)

def make_variants(bgr):
    if bgr is None or bgr.size == 0: return []
    h,w = bgr.shape[:2]
    s = min(h,w); x0,y0 = (w-s)//2, (h-s)//2
    sq = bgr[y0:y0+s, x0:x0+s].copy()
    if s < 1200:
        scale = 1200.0 / s
        sq = cv2.resize(sq, (int(s*scale), int(s*scale)), interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(sq, cv2.COLOR_BGR2GRAY)
    gray = cv2.bitwise_and(gray, _center_circle_mask(gray, 0.92))
    sup   = _suppress_glare(gray)
    sharp = _enhance(sup)
    _, otsu = cv2.threshold(sharp, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    inv    = cv2.bitwise_not(otsu)
    ada    = cv2.adaptiveThreshold(sharp,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,41,5)
    k = cv2.getStructuringElement(cv2.MORPH_RECT,(2,2))
    otsu = cv2.morphologyEx(otsu, cv2.MORPH_OPEN, k, 1)
    ada  = cv2.morphologyEx(ada,  cv2.MORPH_OPEN, k, 1)

    outs=[]
    for g in [sharp, otsu, inv, ada]:
        rgb0 = cv2.cvtColor(g, cv2.COLOR_GRAY2RGB)
        rgb1 = cv2.rotate(rgb0, cv2.ROTATE_180)
        outs.extend([rgb0, rgb1])
    strong = cv2.convertScaleAbs(sharp, alpha=1.6, beta=-15)
    outs.append(cv2.cvtColor(strong, cv2.COLOR_GRAY2RGB))
    return outs

# —— 文本清洗 + 型号提取 —— 
RE_CR_RAW = re.compile(r"([CDGOQ]?R)\s*([0O]\s*\d\s*\d\s*\d|\d\s*\d\s*\d\s*\d)", re.I)

def _normalize_text(t:str)->str:
    s=t.upper()
    s = re.sub(r"\b[DGQO]R","CR",s)  # DR/GR/QR/OR -> CR
    s = (s.replace("O","0").replace("S","5").replace("Z","2")
           .replace("B","8").replace("G","6").replace("I","1").replace("L","1"))
    return re.sub(r"\s+"," ",s).strip()

def extract_models(txt:str):
    txt=_normalize_text(txt)
    cand=set()
    for m in re.finditer(r"CR\s*(\d{4})", txt): cand.add("CR"+m.group(1))
    if not cand:
        for m in RE_CR_RAW.finditer(txt):
            digits=re.sub(r"\s+","",m.group(2))
            if len(digits)==4 and digits.isdigit(): cand.add("CR"+digits)
    allowed=[c for c in cand if c in ALLOWED]
    return allowed or list(cand)

# —— OCR 实现（RapidOCR 优先，Tesseract 备选）——
def rapidocr_read(imgs):
    try:
        from rapidocr_onnxruntime import RapidOCR
    except Exception:
        return []
    ocr=RapidOCR()
    outs=[]
    for i,im in enumerate(imgs):
        try:
            res,_ = ocr(im)  # [(box, text, score), ...]
            if not res: continue
            best_conf=0.0; all_text=[]
            for item in res:
                if len(item)>=3:
                    all_text.append(str(item[1]))
                    sc=float(item[2]) if item[2] is not None else 0.0
                    best_conf=max(best_conf, sc)
            outs.append((" ".join(all_text), best_conf, f"v{i:02d}"))
        except Exception:
            pass
    return outs

def tesseract_read(imgs):
    try:
        import pytesseract
    except Exception:
        return []
    outs=[]
    for i,im in enumerate(imgs):
        gray=cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)
        for psm in (6,7,11,13):
            try:
                txt=pytesseract.image_to_string(gray, config=f"--oem 3 --psm {psm}")
                txt=txt.replace("\n"," ").strip()
                if txt: outs.append((txt, 0.0, f"v{i:02d}-psm{psm}"))
            except Exception:
                pass
    return outs

def choose_best(results):
    # 选择：命中白名单数量多者优先；再看置信度
    best = ("", [], -1.0)
    for txt, conf, _ in results:
        models = extract_models(txt)
        score = (len([m for m in models if m in ALLOWED]) * 2.0) + conf
        if score > best[2]:
            best = (" ".join(models) if models else "", models, score)
    return best  # (model_join, models_list, score)

# —— Worker 入口（供 main 异步进程使用）——
def ocr_worker(input_q, output_q):
    import time, queue
    while True:
        try:
            task = input_q.get(timeout=0.2)
        except queue.Empty:
            continue
        if task is None: break
        tid = task["tid"]; crop = task["crop"]
        t0 = time.time()
        variants = make_variants(crop)
        results = rapidocr_read(variants) or tesseract_read(variants)
        model_join, models, score = choose_best(results)
        ms = (time.time()-t0)*1000.0
        # 返回第一个型号（如果有多个就按出现顺序取第一个）
        model = models[0] if models else ""
        raw = model_join if model_join else ""
        output_q.put({"tid":tid, "model":model, "raw":raw, "prob":score, "ms":ms})
