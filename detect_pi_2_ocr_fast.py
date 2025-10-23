# RPi5 | YOLO + PaddleOCR(英文) | 电池表面OCR（无卡面）| 限额+锁定 | 颜色规则

import os, time, argparse, re
from datetime import datetime
import cv2, numpy as np
from ultralytics import YOLO
from picamera2 import Picamera2

# ---------------- OCR 规则（只抽电池型号） ----------------
# 修复：确保所有变量名和等号之间都使用标准空格
RE_CR     = re.compile(r"CR\s*(1616|1620|2016|2025|2032)", re.I)
RE_DIGITS = re.compile(r"(1616|1620|2016|2025|2032)")

def norm_model(s: str) -> str:
    """标准化 OCR 结果为 CRxxxx 格式的电池型号。"""
    if not s: return ""
    s = s.upper().replace(" ", "")
    # 优先匹配 CRxxxx 格式，然后匹配纯数字
    m = RE_CR.search(s) or RE_DIGITS.search(s)
    return "CR" + m.group(1) if m else ""

# ---------------- CLI ----------------
def parse_args():
    """解析命令行参数。"""
    ap = argparse.ArgumentParser("RPi5 YOLO + PaddleOCR (battery surface only)")
    ap.add_argument("--weights", required=True, help="YOLO 权重（NCNN目录/pt/onnx）")
    ap.add_argument("--imgsz", type=int, default=320)
    ap.add_argument("--conf", type=float, default=0.35)
    ap.add_argument("--min_area", type=int, default=1800, help="最小检测框面积（像素）")
    ap.add_argument("--save_dir", default="/home/pi/hard_cases", help="困难案例图片保存目录")
    ap.add_argument("--headless", action="store_true", help="无头模式（不显示界面）")
    ap.add_argument("--preview", action="store_true", help="使用 Picamera2 预览配置")
    ap.add_argument("--assume_bgr", action="store_true", help="假设 Picamera2 输出已经是 BGR 格式")

    # OCR 调度参数
    ap.add_argument("--ocr", action="store_true", help="启用 PaddleOCR")
    ap.add_argument("--ocr_every", type=int, default=6, help="每 N 帧尝试 OCR 一次")
    ap.add_argument("--ocr_budget", type=int, default=3, help="每帧最多 OCR ROI 数")
    ap.add_argument("--ocr_lock_conf", type=float, default=0.55, help="达到该置信度即锁定型号 (0~1)")
    ap.add_argument("--ocr_pad", type=float, default=0.12, help="ROI 外扩比例")

    # 目标批次（用于上色）
    ap.add_argument("--expected_model", default="", help="期望的电池型号，例如 CR2025")
    ap.add_argument("--expected_version", default="", help="期望的版本，例如 v1/v2/v3；如不区分可不填")
    return ap.parse_args()

# ---------------- 绘制 ----------------
def ensure_dir(p): 
    """确保目录存在。"""
    os.makedirs(p, exist_ok=True)

def draw_box(img, x1,y1,x2,y2, label, color=(255,255,255)):
    """在图像上绘制边界框和标签。"""
    cv2.rectangle(img,(x1,y1),(x2,y2),color,2)
    (tw,th),_ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX,0.6,2)
    y0 = max(0, y1 - th - 6)
    # 绘制背景框
    cv2.rectangle(img,(x1,y0),(x1+tw+8,y0+th+8),color,-1)
    # 绘制文字
    cv2.putText(img,label,(x1+4,y0+th+3),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,0,0),2)

# ---------------- PaddleOCR ----------------
def init_paddle_ocr():
    """初始化 PaddleOCR 实例（只加载英文轻量模型）。"""
    # 只加载英文轻量模型；关闭方向检测
    from paddleocr import PaddleOCR
    # 默认使用 CPU (use_gpu=False)，禁用 cls (cls=False)
    ocr = PaddleOCR(use_textline_orientation=False, lang='en', show_log=False)
    return ocr

def prep_roi_for_ocr(bgr, max_w=240):
    """
    缩放到适中宽度 + 轻微去噪，返回 RGB 图像（PaddleOCR 接受 RGB ndarray）。
    """
    if bgr is None or bgr.size==0: return None
    h,w = bgr.shape[:2]
    # 缩放到适中宽度
    if w > max_w:
        s = max_w/float(w)
        bgr = cv2.resize(bgr, (int(w*s), int(h*s)), interpolation=cv2.INTER_LINEAR)
    
    # 轻微去噪 (3x3 确保不损失太多细节)
    bgr = cv2.GaussianBlur(bgr,(3,3),0)
    # 转换为 RGB 格式
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return rgb

def paddle_ocr_once(ocr, rgb_img):
    """
    执行一次 PaddleOCR。
    返回 (model_code, raw_text, conf)
    conf 取命中型号词条的最高概率；若没有命中，取整句最高概率（作为参考）
    """
    # res 结构: list[list[ [box, (text, prob)], ... ]]
    res = ocr.ocr(rgb_img, cls=False) 
    
    texts = []
    best_prob = 0.0
    best_text = ""
    
    if not res or not res[0]:
        return "", "", 0.0
        
    for box, (txt, prob) in res[0]:
        t = txt.strip()
        texts.append(t)
        
        # 查找是否包含 CRxxxx 格式的型号
        m = RE_CR.search(t) or RE_DIGITS.search(t)
        if m and prob > best_prob:
            best_prob = prob
            best_text = t
            
    raw = " ".join(texts).upper()
    model = norm_model(best_text if best_text else raw)
    
    # 如果找到了型号，使用找到型号对应的最高概率；否则，使用所有检测结果的最高概率作为参考
    conf = float(best_prob if model else max([p for _,(_,p) in res[0]] or [0.0]))
    
    return model, raw, conf

# ---------------- 主程序 ----------------
def main():
    args = parse_args()
    
    # 设置环境变量以优化 NCNN/OpenBLAS 性能（尤其在 RPi 上）
    os.environ["OMP_NUM_THREADS"]="4"; os.environ["NCNN_THREADS"]="4"
    os.environ.setdefault("OPENBLAS_NUM_THREADS","1"); os.environ.setdefault("NCNN_VERBOSE","0")
    ensure_dir(args.save_dir)

    expected_model = norm_model(args.expected_model)
    expected_ver     = args.expected_version.lower().strip() if args.expected_version else ""

    # OCR 初始化
    if args.ocr:
        try:
            ocr = init_paddle_ocr()
            print("✅ PaddleOCR 英文模型初始化成功。")
        except Exception as e:
            print("❌ PaddleOCR 初始化失败：", e)
            args.ocr = False
            ocr = None
    else:
        ocr = None

    # Camera 初始化
    picam2 = Picamera2()
    main_w, main_h = 1280, 960 # 推荐分辨率
    cfg = (picam2.create_preview_configuration if args.preview else picam2.create_video_configuration)(
        main={"size":(main_w,main_h),"format":"YUV420"}, controls={"FrameRate":30}
    )
    picam2.configure(cfg); picam2.start(); time.sleep(1.0)
    print(f"✅ Picamera2 初始化成功，分辨率: {main_w}x{main_h} @ 30FPS")

    # YOLO 初始化
    print("Loading YOLO…")
    model = YOLO(args.weights)
    # 暖机 (Warm-up)
    warm = np.zeros((args.imgsz,args.imgsz,3), np.uint8)
    _ = model.predict(source=warm, imgsz=args.imgsz, verbose=False)
    print("✅ YOLO 模型加载并暖机完毕。")

    # 状态缓存
    # tid -> {"model":str|"" , "conf":float, "locked":bool} 存储每个跟踪目标的状态
    tracks = {}
    # 缩放系数
    sx, sy = main_w/float(args.imgsz), main_h/float(args.imgsz)
    
    # 性能/保存状态
    last_save=0.0; fps_hist=[]; frame_idx=0; start=time.time()
    total_ocr_ms=0.0; ocr_runs=0

    try:
        while True:
            t0=time.time()
            
            # 捕获帧并转换为 BGR
            yuv = picam2.capture_array("main")
            frame = yuv if args.assume_bgr else cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR_I420)

            # YOLO 推理（缩放图像进行推理）
            infer = cv2.resize(frame,(args.imgsz,args.imgsz), interpolation=cv2.INTER_LINEAR)
            r = model.track(source=infer, imgsz=args.imgsz, conf=args.conf, verbose=False, persist=True)[0]

            det=0; low=False; ids=set()
            # 设置 OCR 预算，限制每帧 OCR 数量
            budget = args.ocr_budget if args.ocr else 0
            do_ocr = args.ocr and (frame_idx % args.ocr_every == 0)

            if r.boxes is not None and len(r.boxes)>0 and r.boxes.id is not None:
                # 按置信度排序，高置信度的目标优先进行 OCR
                order = np.argsort((-r.boxes.conf.cpu().numpy()).flatten())
                
                for i in order:
                    b = r.boxes[int(i)]
                    # 归一化坐标 (0-imgsz)
                    x1,y1,x2,y2 = map(int, b.xyxy[0].tolist())
                    # 实际坐标 (0-main_w/h)
                    X1,Y1,X2,Y2 = int(x1*sx),int(y1*sy),int(x2*sx),int(y2*sy)
                    
                    # 面积过滤
                    if (X2-X1)*(Y2-Y1) < args.min_area: continue
                    
                    conf = float(b.conf[0]) if b.conf is not None else 0.0
                    det += 1
                    # 检查是否有低置信度检测
                    if conf < (args.conf+0.10): low=True

                    tid = int(b.id[0]); ids.add(tid)
                    info = tracks.get(tid, {"model":"", "conf":0.0, "locked":False})

                    # OCR 调度逻辑
                    if do_ocr and (not info["locked"]) and budget>0 and ocr is not None:
                        # 计算 ROI 外扩区域
                        pad=args.ocr_pad; w,h = X2-X1, Y2-Y1
                        px,py = int(w*pad), int(h*pad)
                        # 确保不越界
                        ox1,oy1 = max(0, X1-px), max(0, Y1-py)
                        ox2,oy2 = min(main_w, X2+px), min(main_h, Y2+py)
                        crop = frame[oy1:oy2, ox1:ox2]

                        # OCR 预处理和执行
                        rgb = prep_roi_for_ocr(crop, max_w=240)
                        if rgb is not None:
                            t_ocr0=time.time()
                            m, raw, prob = paddle_ocr_once(ocr, rgb)
                            ocr_ms=(time.time()-t_ocr0)*1000.0
                            total_ocr_ms+=ocr_ms; ocr_runs+=1; budget-=1

                            # 更新最佳 OCR 结果
                            if m and (prob>info["conf"] or not info["model"]):
                                info["model"]=m
                                info["conf"]=prob
                                # 达到锁定置信度
                                if prob >= args.ocr_lock_conf:
                                    info["locked"]=True

                    # 上色与标签逻辑
                    label_model = info["model"] if info["model"] else "?"
                    
                    # 颜色：默认白（未知/未设批次）
                    color = (255,255,255)  
                    if expected_model:
                        if info["model"] == expected_model:
                            # 绿 = 完全匹配 (V2 版本判断可在此添加)
                            color = (0,255,0)   
                        elif info["model"]:      
                            # 红 = 型号不符
                            color = (0,0,255)   
                        else:
                            # 白 = 未读出（未读出不误判红）
                            color = (255,255,255)  

                    # 绘制标签
                    label = f"{label_model} | {conf:.2f}"
                    # 额外的锁定标记
                    if info["locked"]: label += " L"
                    draw_box(frame, X1,Y1,X2,Y2, label, color)
                    tracks[tid]=info

            # 清理离场（移除未出现的跟踪 ID）
            for tid in [t for t in list(tracks.keys()) if t not in ids]:
                tracks.pop(tid, None)

            # HUD 性能数据显示
            frame_idx+=1
            dt=(time.time()-t0)*1000.0
            fps_hist.append(1000.0/max(dt,1.0))
            if len(fps_hist)>30: fps_hist.pop(0)
            fps=sum(fps_hist)/len(fps_hist)
            
            hud=f"检测:{det} | 耗时:{dt:.1f}ms | FPS:{fps:.1f}"
            if args.ocr and ocr_runs:
                hud+=f" | OCR平均:{(total_ocr_ms/ocr_runs):.1f}ms"
            cv2.putText(frame, hud, (10,28), cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,255,255),2)

            # 显示窗口
            if not args.headless:
                cv2.imshow("电池检测 + PaddleOCR", frame)
                if cv2.waitKey(1)&0xFF==27: break

            # 困难案例保存 (无检测 或 置信度低)
            now=time.time()
            if (det==0 or low) and (now-last_save>1.0):
                fn=f"hard_{det}_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.jpg"
                cv2.imwrite(os.path.join(args.save_dir, fn), frame); last_save=now

    finally:
        # 清理资源
        try: picam2.stop()
        except: pass
        if not args.headless: cv2.destroyAllWindows()
        el=time.time()-start
        print(f"\n--- 性能摘要 ---")
        print(f"总帧数:{frame_idx} | 总耗时:{el:.1f}s | 平均 FPS:{frame_idx/max(el,1):.1f}")
        if args.ocr and ocr_runs:
            print(f"OCR 运行次数:{ocr_runs} | 平均 OCR 耗时:{total_ocr_ms/ocr_runs:.1f} ms")
        print("程序结束。")

if __name__=="__main__":
    main()
