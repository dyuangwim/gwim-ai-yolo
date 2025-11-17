import os, sys, json, signal, subprocess
from datetime import datetime
from PyQt5 import QtCore, QtGui, QtWidgets, uic

# 仅用于“Stop/关闭窗口时”的最终保险；Stop Alarm 不触GPIO
try:
    from utils_hw import Buzzer as SafeBuzzer
except Exception:
    SafeBuzzer = None

BATTERY_OPTIONS = [1, 2, 4, 6, 8, 10, 12, 16, 20, 24]

class ResultCard(QtWidgets.QFrame):
    clicked = QtCore.pyqtSignal(dict)
    def __init__(self, info, parent=None):
        super().__init__(parent); self.info=info; self.setObjectName("ResultCard"); self.setFrameShape(QtWidgets.QFrame.NoFrame)
        border_color = "#22c55e" if info.get("ng_count",0)==0 else "#ef4444"
        self.setStyleSheet(f"QFrame#ResultCard{{border-radius:12px;border:2px solid {border_color};background:transparent;}}")
        root = QtWidgets.QVBoxLayout(self); root.setContentsMargins(2,2,2,2); root.setSpacing(0)
        imgf = QtWidgets.QFrame(); imgf.setStyleSheet("background:#020617;border-radius:10px;")
        il = QtWidgets.QVBoxLayout(imgf); il.setContentsMargins(4,4,4,4)
        self.imageLabel = QtWidgets.QLabel(alignment=QtCore.Qt.AlignCenter); self.imageLabel.setMinimumSize(220,120)
        self.imageLabel.setStyleSheet("background:#020617;border-radius:8px;color:#64748b;"); il.addWidget(self.imageLabel); root.addWidget(imgf,1)
        bottom = QtWidgets.QFrame(); bottom.setStyleSheet("background:white;border-radius:10px;")
        bl = QtWidgets.QVBoxLayout(bottom); bl.setContentsMargins(8,4,8,4); bl.setSpacing(2)
        ts = info.get("ts"); ts_text = ts.strftime("%H:%M:%S") if isinstance(ts, datetime) else str(ts)
        t = QtWidgets.QLabel(ts_text); t.setStyleSheet("color:#0f172a;font-weight:bold;"); bl.addWidget(t)
        expected, ng, pc = info.get("expected",0), info.get("ng_count",0), info.get("pack_count",0)
        s = f"Detected: {expected} batteries/pkg | Packs: {pc} | NG: {ng}" if pc else f"NG packs: {ng}"
        x = QtWidgets.QLabel(s); x.setStyleSheet("color:#0f172a;"); bl.addWidget(x); root.addWidget(bottom,0)
        badge = QtWidgets.QLabel("  Pass  " if ng==0 else "  Fail  ", self)
        badge.setStyleSheet("background:%s;color:white;border-radius:10px;font-weight:bold;padding:2px 6px;" % ("#22c55e" if ng==0 else "#ef4444"))
        badge.move(self.width()-badge.width()-8,8); badge.raise_(); self._badge=badge
        if info.get("image_path") and os.path.exists(info["image_path"]):
            p = QtGui.QPixmap(info["image_path"]); 
            if not p.isNull(): self.set_image(p)
    def resizeEvent(self, e): super().resizeEvent(e); self._badge.move(self.width()-self._badge.width()-8,8)
    def set_image(self, p): size=self.imageLabel.size() or QtCore.QSize(220,120); self.imageLabel.setPixmap(p.scaled(size, QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation))
    def mousePressEvent(self, e): 
        if e.button()==QtCore.Qt.LeftButton: self.clicked.emit(self.info)
        super().mousePressEvent(e)

class DetailDialog(QtWidgets.QDialog):
    def __init__(self, info, parent=None):
        super().__init__(parent); self.setWindowTitle("Inspection Details"); self.resize(520,520); self.setModal(True)
        self.setStyleSheet("QDialog{background:#f9fafb;}")
        lay = QtWidgets.QVBoxLayout(self); lay.setContentsMargins(16,16,16,16); lay.setSpacing(12)
        hdr = QtWidgets.QHBoxLayout(); t = QtWidgets.QLabel("Inspection Details"); t.setStyleSheet("font-size:18px;font-weight:bold;"); hdr.addWidget(t)
        status = QtWidgets.QLabel("  Pass  " if info.get("ng_count",0)==0 else "  Fail  ")
        status.setStyleSheet("background:%s;color:white;border-radius:12px;font-weight:bold;padding:4px 8px;" % ("#22c55e" if info.get("ng_count",0)==0 else "#ef4444"))
        hdr.addWidget(status); hdr.addStretch(); lay.addLayout(hdr)
        ts = info.get("ts"); ts_label = QtWidgets.QLabel((ts.strftime("%Y-%m-%d %H:%M:%S") if isinstance(ts, datetime) else str(ts))); ts_label.setStyleSheet("color:#6b7280;"); lay.addWidget(ts_label)
        imgf = QtWidgets.QFrame(); imgf.setStyleSheet("background:white;border-radius:12px;"); il = QtWidgets.QVBoxLayout(imgf); il.setContentsMargins(8,8,8,8)
        img = QtWidgets.QLabel(alignment=QtCore.Qt.AlignCenter); img.setMinimumSize(400,260); img.setStyleSheet("background:#0b1120;border-radius:10px;"); il.addWidget(img); lay.addWidget(imgf)
        if info.get("image_path") and os.path.exists(info["image_path"]):
            p = QtGui.QPixmap(info["image_path"]); 
            if not p.isNull(): img.setPixmap(p.scaled(QtCore.QSize(480,320), QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation))
        inf = QtWidgets.QFrame(); inf.setStyleSheet("background:white;border-radius:12px;"); grid = QtWidgets.QGridLayout(inf); grid.setContentsMargins(12,8,12,8); grid.setHorizontalSpacing(32)
        exp, ng, pc = info.get("expected",0), info.get("ng_count",0), info.get("pack_count",0)
        labels=[("Expected per pack", f"{exp} batteries"), ("Total packs", str(pc)), ("NG packs", str(ng))]
        for i,(k,v) in enumerate(labels):
            a=QtWidgets.QLabel(k); a.setStyleSheet("color:#6b7280;"); b=QtWidgets.QLabel(v); b.setStyleSheet("font-weight:bold;"+("color:#22c55e;" if k=="NG packs" and ng==0 else "")); grid.addWidget(a,0,i); grid.addWidget(b,1,i)
        lay.addWidget(inf)
        box = QtWidgets.QFrame(); box.setStyleSheet("background:%s;border-radius:12px;" % ("#dcfce7" if ng==0 else "#fee2e2"))
        bl = QtWidgets.QVBoxLayout(box); bl.setContentsMargins(12,8,12,8); title = QtWidgets.QLabel("Status"); title.setStyleSheet("font-weight:bold;"); bl.addWidget(title)
        msg = "All packs in this image matched the expected quantity." if ng==0 else f"Battery count mismatch detected in {ng} pack(s). Please check before continuing."
        body = QtWidgets.QLabel(msg); body.setWordWrap(True); bl.addWidget(body); lay.addWidget(box)
        btn = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Close); btn.rejected.connect(self.reject); lay.addWidget(btn)

class LogReaderThread(QtCore.QThread):
    newResult = QtCore.pyqtSignal(dict); processExited = QtCore.pyqtSignal(int)
    def __init__(self, process: subprocess.Popen, expected_per_pack: int, parent=None):
        super().__init__(parent); self._process=process; self._expected=expected_per_pack
    def run(self):
        p=self._process; current=None; jpath=None
        while True:
            line=p.stdout.readline()
            if not line: break
            s=line.strip()
            if s.startswith("Image: "): current = s.split("Image:",1)[1].strip()
            elif s.startswith("JSON:"): jpath = s.split("JSON:",1)[1].strip(); info=self._build_info(current,jpath); 
            if jpath:
                if info: self.newResult.emit(info)
                current=None; jpath=None
        p.wait(); self.processExited.emit(p.returncode)
    def _build_info(self, image_path, json_path):
        if not image_path: return None
        info={"image_path":image_path,"expected":self._expected,"ts":datetime.now(),"ng_count":0,"pack_count":0,"packs":[],"json_path":json_path}
        if json_path and os.path.exists(json_path):
            try:
                with open(json_path,"r") as f: data=json.load(f)
                packs=data.get("packs",[]); info["packs"]=packs; info["pack_count"]=len(packs); info["ng_count"]=len([p for p in packs if not p.get("ok",True)])
            except Exception as e:
                print("Failed to parse JSON:", e, file=sys.stderr)
        return info

class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        uic.loadUi(os.path.join(os.path.dirname(__file__),"batch_inspector.ui"), self)
        self.setStyleSheet("""
        QMainWindow{background:#f1f5f9;} QLabel{font-size:14px;} QComboBox,QSpinBox{font-size:14px;}
        QFrame#controlCard,QFrame#stats_card,QFrame#results_card{background:white;border-radius:16px;}
        QFrame#how_frame{background:#eff6ff;border-radius:10px;} QFrame#warning_frame{background:#fee2e2;border-radius:10px;}
        QLabel#label_cp_title{font-size:16px;font-weight:bold;} QLabel#title_main{font-size:20px;font-weight:bold;}
        QLabel#title_sub{color:#6b7280;} QLabel#empty_label{color:#9ca3af;}
        """)
        self.combo_expected.setStyleSheet("QComboBox{padding:8px 10px;border-radius:8px;border:1px solid #cbd5e1;background:white;}")
        self.btn_start.setStyleSheet("QPushButton{background:#16a34a;color:white;font-size:16px;font-weight:bold;border:none;border-radius:10px;} QPushButton:disabled{background:#9ca3af;}")
        self.btn_stop.setStyleSheet("QPushButton{background:#ef4444;color:white;font-size:15px;font-weight:bold;border:none;border-radius:10px;} QPushButton:disabled{background:#fecaca;color:#7f1d1d;}")
        self.btn_stop_alarm.setStyleSheet("QPushButton{background:#b91c1c;color:white;font-size:15px;font-weight:bold;border:none;border-radius:10px;} QPushButton:disabled{background:#fee2e2;color:#991b1b;}")
        self.btn_reset.setStyleSheet("QPushButton{background:white;color:#111827;font-size:14px;border:1px solid #e5e7eb;border-radius:10px;}")
        self.warning_title.setStyleSheet("font-weight:bold;color:#b91c1c;"); self.warning_msg.setStyleSheet("color:#b91c1c;"); self.how_title.setStyleSheet("font-weight:bold;")
        self.stat_expected_frame.setStyleSheet("background:#f9fafb;border-radius:12px;")
        self.stat_total_frame.setStyleSheet("background:#eff6ff;border-radius:12px;")
        self.stat_passed_frame.setStyleSheet("background:#ecfdf3;border-radius:12px;")
        self.stat_failed_frame.setStyleSheet("background:#fef2f2;border-radius:12px;")
        for x in [self.label_stat_expected,self.label_stat_total,self.label_stat_passed,self.label_stat_failed]: x.setStyleSheet("color:#6b7280;")
        for x in [self.stat_expected_val,self.stat_total_val,self.stat_passed_val,self.stat_failed_val]: x.setStyleSheet("font-size:20px;font-weight:bold;")
        for x in [self.label_stat_expected_unit,self.label_stat_total_unit,self.label_stat_passed_unit,self.label_stat_failed_unit]: x.setStyleSheet("color:#9ca3af;")

        self.combo_expected.clear()
        for n in BATTERY_OPTIONS: self.combo_expected.addItem(f"{n} Battery" if n==1 else f"{n} Batteries", n)
        self.combo_expected.setCurrentIndex(2)

        self.proc=None; self.log_thread=None
        self.total_inspected=self.total_pass=self.total_fail=0
        self.cards_layout=self.findChild(QtWidgets.QGridLayout,"cardsLayout")

        self.btn_start.clicked.connect(self.start_inspection)
        self.btn_stop.clicked.connect(self.stop_inspection)
        self.btn_stop_alarm.clicked.connect(self.stop_alarm)
        self.btn_reset.clicked.connect(self.reset_counters)
        self.combo_expected.currentIndexChanged.connect(self.update_expected_stat)

        self.update_expected_stat()

    # --- 控制 ---
    def update_expected_stat(self):
        self.stat_expected_val.setText(str(int(self.combo_expected.currentData())))

    def _update_pass_rate_label(self):
        rate=0.0 if self.total_inspected==0 else self.total_pass*100.0/self.total_inspected
        self.label_stat_passed_unit.setText(f"{rate:.1f}% rate")

    def start_inspection(self):
        if self.proc is not None: return
        expected=int(self.combo_expected.currentData()); self.update_expected_stat()
        cmd=["python3","/home/pi/battery_batch/auto_capture.py","--expected",str(expected),"--buzzer_pin","21","--keep_raw","200","--keep_out","500"]
        try:
            self.proc=subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, stdin=subprocess.PIPE, text=True, bufsize=1)
        except FileNotFoundError:
            QtWidgets.QMessageBox.critical(self,"Error","auto_capture.py not found at /home/pi/battery_batch/"); self.proc=None; return
        self.log_thread=LogReaderThread(self.proc, expected); self.log_thread.newResult.connect(self.on_new_result); self.log_thread.processExited.connect(self.on_process_exited); self.log_thread.start()
        self.btn_start.setEnabled(False); self.combo_expected.setEnabled(False); self.btn_stop.setEnabled(True); self.warning_frame.setVisible(False)

    def _failsafe_gpio_off(self):
        # 只在 Stop/关闭窗口调用；Stop Alarm 不触GPIO，避免与子进程竞态
        try:
            if SafeBuzzer is not None:
                b=SafeBuzzer(pin=21, active_high=True)
                b.off(); b.close()
        except Exception:
            pass

    def stop_inspection(self):
        if self.proc is None: return
        try:
            try:
                if self.proc.stdin: self.proc.stdin.write("\n"); self.proc.stdin.flush()
            except Exception: pass
            try: self.proc.send_signal(signal.SIGINT)
            except Exception: pass
            try: self.proc.wait(timeout=1.2)
            except subprocess.TimeoutExpired:
                try: self.proc.terminate()
                except Exception: pass
                try: self.proc.wait(timeout=1.2)
                except subprocess.TimeoutExpired:
                    try: self.proc.kill()
                    except Exception: pass
            self._failsafe_gpio_off()
        finally:
            try:
                if self.proc and self.proc.stdin: self.proc.stdin.close()
            except Exception: pass
            self.proc=None
            self.btn_start.setEnabled(True); self.combo_expected.setEnabled(True)
            self.btn_stop.setEnabled(False); self.btn_stop_alarm.setEnabled(False); self.warning_frame.setVisible(False)

    def stop_alarm(self):
        if self.proc is None: return
        # 仅通知子进程停报警（stdin 回车），不触GPIO，避免把下一次报警“关死”
        try:
            self.proc.stdin.write("\n"); self.proc.stdin.flush()
        except Exception: pass
        self.btn_stop_alarm.setEnabled(False); self.warning_frame.setVisible(False)

    def reset_counters(self):
        self.total_inspected=self.total_pass=self.total_fail=0
        self.stat_total_val.setText("0"); self.stat_passed_val.setText("0"); self.stat_failed_val.setText("0")
        self._update_pass_rate_label()
        while self.cards_layout.count():
            it=self.cards_layout.takeAt(0); w=it.widget()
            if w: w.deleteLater()
        self.empty_label.setVisible(True)

    @QtCore.pyqtSlot(dict)
    def on_new_result(self, info):
        self.total_inspected += 1
        if info.get("ng_count",0)==0: self.total_pass += 1
        else: self.total_fail += 1
        self.stat_total_val.setText(str(self.total_inspected)); self.stat_passed_val.setText(str(self.total_pass)); self.stat_failed_val.setText(str(self.total_fail)); self._update_pass_rate_label()
        if info.get("ng_count",0)>0: self.warning_frame.setVisible(True); self.btn_stop_alarm.setEnabled(True)
        else: self.warning_frame.setVisible(False)
        row=(self.total_inspected-1)//3; col=(self.total_inspected-1)%3
        card=ResultCard(info); card.clicked.connect(self.show_detail_dialog); self.cards_layout.addWidget(card,row,col); self.empty_label.setVisible(False)

    @QtCore.pyqtSlot(int)
    def on_process_exited(self, code):
        self.proc=None; self.log_thread=None
        self.btn_start.setEnabled(True); self.combo_expected.setEnabled(True)
        self.btn_stop.setEnabled(False); self.btn_stop_alarm.setEnabled(False); self.warning_frame.setVisible(False)

    def show_detail_dialog(self, info):
        DetailDialog(info, self).exec_()

    def closeEvent(self, e):
        try: self.stop_inspection()
        except Exception: pass
        e.accept()

def main():
    app=QtWidgets.QApplication(sys.argv); win=MainWindow(); win.show(); sys.exit(app.exec_())

if __name__ == "__main__":
    main()
