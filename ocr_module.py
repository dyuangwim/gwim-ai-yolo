import re, os, cv2, numpy as np
from typing import List, Tuple, Optional

ALLOWED = {"CR1616","CR1620","CR2016","CR2025","CR2032"}
RE_CR_RAW = re.compile(r"([CDQO]?R)\s*([0O]\s*\d\s*\d\s*\d|\d\s*\d\s*\d\s*\d)", re.I)

def _center_circle_mask(img, ratio=0.92):
    h,w = img.shape[:2]
    r = int(min(h,w)*ratio/2)
    m = np.zeros((h,w), np.uint8)
    cv2.circle(m, (w//2, h//2), r, 255, -1)
    return m

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

def _normalize_text(t:str)->str:
    s = t.upper()
    s = re.sub(r"\b[DGQO]R","CR",s)
    s = (s.replace("O","0").replace("S","5").replace("Z","2").replace("B","8"))
    return re.sub(r"\s+"," ",s).strip()

def extract_models(txt:str)->List[str]:
    txt=_normalize_text(txt)
    cand=set()
    for m in re.finditer(r"CR\s*(\d{4})", txt):
        cand.add("CR"+m.group(1))
    if not cand:
        for m in RE_CR_RAW.finditer(txt):
            digits=re.sub(r"\s+","",m.group(2))
            if len(digits)==4 and digits.isdigit():
                cand.add("CR"+digits)
    return sorted(cand, key=lambda x: (x not in ALLOWED, x))

def make_variants(bgr, max_vars:int=4):
    """返回少量高价值变体：sharp / otsu / inv / strong + 180° 的 sharp。"""
    outs=[]
    if bgr is None or bgr.size==0: return outs
    h,w=bgr.shape[:2]
    s=min(h,w); x0,y0=(w-s)//2,(h-s)//2
    sq=bgr[y0:y0+s, x0:x0+s].copy()
    if s<720:
        k=720.0/s; sq=cv2.resize(sq,(int(s*k),int(s*k)),cv2.INTER_CUBIC)
    gray=cv2.cvtColor(sq, cv2.COLOR_BGR2GRAY)
    gray=cv2.bitwise_and(gray,_center_circle_mask(gray,0.92))
    sup=_suppress_glare(gray); sharp=_enhance(sup)
    _,otsu=cv2.threshold(sharp,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    inv=cv2.bitwise_not(otsu)
    strong=cv2.convertScaleAbs(sharp,alpha=1.6,beta=-15)

    base=[sharp,otsu,inv,strong][:max_vars]
    outs=[cv2.cvtColor(x,cv2.COLOR_GRAY2RGB) for x in base]
    # 只加一个 180 度
    outs.append(cv2.rotate(cv2.cvtColor(sharp,cv2.COLOR_GRAY2RGB), cv2.ROTATE_180))
    return outs

# 单例 RapidOCR
_RAPID=None
def _get_rapid():
    global _RAPID
    if _RAPID is None:
        try:
            from rapidocr_onnxruntime import RapidOCR
            _RAPID = RapidOCR()
        except Exception:
            _RAPID=False
    return _RAPID

def rapidocr_read(imgs, early_stop=True)->List[Tuple[str,float,str]]:
    ocr=_get_rapid()
    if not ocr: return []
    outs=[]
    for i,im in enumerate(imgs):
        try:
            res,_=ocr(im)
            if not res: continue
            best=0.0; buf=[]
            for item in res:
                if len(item)>=3:
                    buf.append(str(item[1]))
                    sc=float(item[2]) if item[2] is not None else 0.0
                    best=max(best,sc)
            text=" ".join(buf)
            outs.append((text,best,f"v{i:02d}"))
            if early_stop and extract_models(_normalize_text(text)) and best>=0.55:
                break
        except Exception:
            pass
    return outs

def tesseract_read(imgs)->List[Tuple[str,float,str]]:
    try:
        import pytesseract
    except Exception:
        return []
    outs=[]
    for i,im in enumerate(imgs):
        gray=cv2.cvtColor(im,cv2.COLOR_RGB2GRAY)
        best_txt=""; any_added=False
        for psm in (6,7,13):  # 精简 psm 组合提速
            try:
                txt=pytesseract.image_to_string(gray,config=f"--oem 3 --psm {psm}")
                txt=txt.replace("\n"," ").strip()
                if txt:
                    best_txt = best_txt+" "+txt if best_txt else txt
                    any_added=True
            except Exception: pass
        if any_added:
            outs.append((best_txt,0.30,f"v{i:02d}-tess"))
    return outs

def choose_best(results, expected: Optional[str]=None):
    best=("",[], -1.0)
    for txt, conf, _ in results:
        models=extract_models(txt)
        score=(len([m for m in models if m in ALLOWED]) * 1.0) \
              + (0.5 if (expected and expected in models) else 0.0) \
              + conf
        if score>best[2]:
            best=(" ".join(models) if models else "", models, conf)
    return best

def ocr_worker(input_q, output_q):
    import time, queue
    _=_get_rapid()
    while True:
        try:
            task=input_q.get(timeout=0.2)
        except queue.Empty:
            continue
        if task is None: break

        tid=task["tid"]; crop=task["crop"]
        expected=task.get("expected","")
        pre_dir=task.get("pre_dir","")
        max_vars=int(task.get("max_vars",4))
        use_tess=bool(task.get("use_tess",True))
        dump_token=task.get("dump_token","")

        t0=time.time()
        variants=make_variants(crop, max_vars=max_vars)

        # 可选地保存预处理图
        if pre_dir:
            os.makedirs(pre_dir, exist_ok=True)
            for idx,img in enumerate(variants):
                cv2.imwrite(os.path.join(
                    pre_dir, f"pre_{dump_token}_v{idx:02d}.jpg"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

        results=rapidocr_read(variants) or (tesseract_read(variants) if use_tess else [])
        join, models, best_conf = choose_best(results, expected=expected)
        ms=(time.time()-t0)*1000.0
        model=models[0] if models else ""
        output_q.put({"tid":tid,"model":model,"raw":join or "","prob":float(best_conf),"ms":ms})
