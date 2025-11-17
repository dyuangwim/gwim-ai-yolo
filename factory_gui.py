import os, sys, json, signal, subprocess
from datetime import datetime
from PyQt5 import QtCore, QtGui, QtWidgets, uic

# 最后保险（不主动使用，只在极端情况下拉低）
try:
    from utils_hw import Buzzer as SafeBuzzer
except Exception:
    SafeBuzzer = None

BATTERY_OPTIONS = [1,2,4,6,8,10,12,16,20,24]

class ResultCard(QtWidgets.QFrame):
    clicked = QtCore.pyqtSignal(dict)
    def __init__(self, info, parent=None):
        super().__init__(parent); self.info = info
        self.setObjectName("ResultCard"); self.setFrameShape(QtWidgets.QFrame.NoFrame)
        border = "#22c55e" if info.get("ng_count",0)==0 else "#ef4444"
        self.setStyleSheet(f"QFrame#ResultCard{{border-radius:12px;border:2px solid {border};background:transparent;}}")
        root = QtWidgets.QVBoxLayout(self); root.setContentsMargins(2,2,2,2); root.setSpacing(0)
        imgf = QtWidgets.QFrame(); imgf.setStyleSheet("background:#020617;border-radius:10px;")
        imgl = QtWidgets.QVBoxLayout(imgf); imgl.setContentsMargins(4,4,4,4)
        self.imageLabel = QtWidgets.QLabel(); self.imageLabel.setAlignment(QtCore.Qt.AlignCenter)
        self.imageLabel.setMinimumSize(220,120)
        self.imageLabel.setStyleSheet("background:#020617;border-radius:8px;color:#64748b;")
        imgl.addWidget(self.imageLabel); root.addWidget(imgf,1)
        bot = QtWidgets.QFrame(); bot.setStyleSheet("background:white;border-radius:10px;")
        botl = QtWidgets.QVBoxLayout(bot); botl.setContentsMargins(8,4,8,4); botl.setSpacing(2)
        ts = info.get("ts"); ts_text = ts.strftime("%H:%M:%S") if isinstance(ts, datetime) else str(ts)
        self.timeLabel = QtWidgets.QLabel(ts_text); self.timeLabel.setStyleSheet("color:#0f172a;font-weight:bold;")
        botl.addWidget(self.timeLabel)
        expected = info.get("expected",0); ng = info.get("ng_count",0); pc = info.get("pack_count",0)
        text = f"Detected: {expected} batteries/pkg | Packs: {pc} | NG: {ng}" if pc else f"NG packs: {ng}"
        self.summaryLabel = QtWidgets.QLabel(text); self.summaryLabel.setStyleSheet("color:#0f172a;"); botl.addWidget(self.summaryLabel)
        root.addWidget(bot,0)
        badge = QtWidgets.QLabel("  Pass  " if ng==0 else "  Fail  ", self)
        badge.setAlignment(QtCore.Qt.AlignCenter)
        badge.setStyleSheet("background:%s;color:white;border-radius:10px;font-weight:bold;padding:2px 6px;"
                            % ("#22c55e" if ng==0 else "#ef4444"))
        badge.move(self.width()-badge.width()-8,8); badge.raise_(); self._badge = badge
        if info.get("image_path") and os.path.exists(info["image_path"]):
            pix = QtGui.QPixmap(info["image_path"]); 
            if not pix.isNull(): self.set_image(pix)
    def resizeEvent(self,e): super().resizeEvent(e); self._badge.move(self.width()-self._badge.width()-8,8)
    def set_image(self,pix):
        size = self.imageLabel.size(); 
        if size.width()<=0 or size.height()<=0: size = QtCore.QSize(220,120)
        self.imageLabel.setPixmap(pix.scaled(size, QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation))
    def mousePressEvent(self,e):
        if e.button()==QtCore.Qt.LeftButton: self.clicked.emit(self.info)
        super().mousePressEvent(e)

class DetailDialog(QtWidgets.QDialog):
    def __init__(self, info, parent=None):
        super().__init__(parent); self.setWindowTitle("Inspection Details"); self.resize(520,520)
        self.setModal(True); self.setStyleSheet("QDialog{background:#f9fafb;}")
        lay = QtWidgets.QVBoxLayout(self); lay.setContentsMargins(16,16,16,16); lay.setSpacing(12)
        h = QtWidgets.QHBoxLayout(); t = QtWidgets.QLabel("Inspection Details"); t.setStyleSheet("font-size:18px;font-weight:bold;"); h.addWidget(t)
        s = QtWidgets.QLabel("  Pass  " if info.get("ng_count",0)==0 else "  Fail  ")
        s.setStyleSheet("background:%s;color:white;border-radius:12px;font-weight:bold;padding:4px 8px;"
                        % ("#22c55e" if info.get("ng_count",0)==0 else "#ef4444")); h.addWidget(s); h.addStretch(); lay.addLayout(h)
        ts = info.get("ts"); ts_text = ts.strftime("%Y-%m-%d %H:%M:%S") if isinstance(ts, datetime) else str(ts)
        ts_lbl = QtWidgets.QLabel(ts_text); ts_lbl.setStyleSheet("color:#6b7280;"); lay.addWidget(ts_lbl)
        imgf = QtWidgets.QFrame(); imgf.setStyleSheet("background:white;border-radius:12px;")
        imgl = QtWidgets.QVBoxLayout(imgf); imgl.setContentsMargins(8,8,8,8)
        self.imageLabel = QtWidgets.QLabel(); self.imageLabel.setAlignment(QtCore.Qt.AlignCenter)
        self.imageLabel.setMinimumSize(400,260); self.imageLabel.setStyleSheet("background:#0b1120;border-radius:10px;")
        imgl.addWidget(self.imageLabel); lay.addWidget(imgf)
        if info.get("image_path") and os.path.exists(info["image_path"]):
            pix = QtGui.QPixmap(info["image_path"])
            if not pix.isNull():
                self.imageLabel.setPixmap(pix.scaled(QtCore.QSize(480,320), QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation))
        infof = QtWidgets.QFrame(); infof.setStyleSheet("background:white;border-radius:12px;")
        grid = QtWidgets.QGridLayout(infof); grid.setContentsMargins(12,8,12,8); grid.setHorizontalSpacing(32)
        e = info.get("expected",0); ng = info.get("ng_count",0); pc = info.get("pack_count",0)
        def lbl(txt,style=""): w=QtWidgets.QLabel(txt); 
        lbl1 = QtWidgets.QLabel("Expected per pack"); lbl1.setStyleSheet("color:#6b7280;")
        val1 = QtWidgets.QLabel(f"{e} batteries"); val1.setStyleSheet("font-weight:bold;")
        lbl2 = QtWidgets.QLabel("Total packs"); lbl2.setStyleSheet("color:#6b7280;")
        val2 = QtWidgets.QLabel(str(pc)); val2.setStyleSheet("font-weight:bold;")
        lbl3 = QtWidgets.QLabel("NG packs"); lbl3.setStyleSheet("color:#6b7280;")
        val3 = QtWidgets.QLabel(str(ng)); val3.setStyleSheet("font-weight:bold;color:%s;"%("#22c55e" if ng==0 else "#ef4444"))
        grid.addWidget(lbl1,0,0); grid.addWidget(val1,1,0); grid.addWidget(lbl2,0,1); grid.addWidget(val2,1,1); grid.addWidget(lbl3,0,2); grid.addWidget(val3,1,2)
        lay.addWidget(infof)
        issuef = QtWidgets.QFrame(); issuef.setStyleSheet("background:%s;border-radius:12px;"%("#dcfce7" if ng==0 else "#fee2e2"))
        v = QtWidgets.QVBoxLayout(issuef); v.setContentsMargins(12,8,12,8)
        ttl = QtWidgets.QLabel("Status"); ttl.setStyleSheet("font-weight:bold;"); v.addWidget(ttl)
        msg = "All packs in this image matched the expected quantity." if ng==0 else f"Battery count mismatch in {ng} pack(s). Please check."
        body = QtWidgets.QLabel(msg); body.setWordWrap(True); v.addWidget(body); lay.addWidget(issuef)
        btn = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Close); btn.rejected.connect(self.reject); lay.addWidget(btn)

class LogReaderThread(QtCore.QThread):
    newResult = QtCore.pyqtSignal(dict); processExited = QtCore.pyqtSignal(int)
    def __init__(self, process: subprocess.Popen, expected_per_pack: int, parent=None):
        super().__init__(parent); self._process = process; self._expected = expected_per_pack
    def run(self):
        p = self._process; cur_img=None; jpath=None
        while True:
            line = p.stdout.readline()
            if not line: break
            line = line.strip()
            if line.startswith("Image: "): cur_img = line.split("Image:",1)[1].strip()
            elif line.startswith("JSON:"):
                jpath = line.split("JSON:",1)[1].strip()
                info = self._build_info(cur_img, jpath)
                if info: self.newResult.emit(info)
                cur_img=None; jpath=None
        p.wait(); self.processExited.emit(p.returncode)
    def _build_info(self, img, j):
        if not img: return None
        info = {"image_path": img, "expected": self._expected, "ts": datetime.now(),
                "ng_count":0, "pack_count":0, "packs":[], "json_path": j}
        if j and os.path.exists(j):
            try:
                with open(j,"r") as f: data=json.load(f)
                packs = data.get("packs", [])
                info["packs"]=packs; info["pack_count"]=len(packs)
                info["ng_count"]=len([p for p in packs if not p.get("ok",True)])
            except Exception as e:
                print("Failed to parse JSON:", e, file=sys.stderr)
        return info

class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        uic.loadUi(os.path.join(os.path.dirname(__file__), "batch_inspector.ui"), self)

        # 样式（同前）
        self.setStyleSheet("""
        QMainWindow{background:#f1f5f9;}
        QLabel{font-size:14px;} QComboBox{font-size:14px;}
        QFrame#controlCard,QFrame#stats_card,QFrame#results_card{background:white;border-radius:16px;}
        QFrame#how_frame{background:#eff6ff;border-radius:10px;}
        QFrame#warning_frame{background:#fee2e2;border-radius:10px;}
        QLabel#label_cp_title{font-size:16px;font-weight:bold;}
        QLabel#title_main{font-size:20px;font-weight:bold;} QLabel#title_sub{color:#6b7280;}
        QLabel#empty_label{color:#9ca3af;}
        """)
        self.combo_expected.setStyleSheet("QComboBox{padding:8px 10px;border-radius:8px;border:1px solid #cbd5e1;background:white;}")
        self.btn_start.setStyleSheet("QPushButton{background:#16a34a;color:white;font-size:16px;font-weight:bold;border:none;border-radius:10px;} QPushButton:disabled{background:#9ca3af;}")
        self.btn_stop.setStyleSheet("QPushButton{background:#ef4444;color:white;font-size:15px;font-weight:bold;border:none;border-radius:10px;} QPushButton:disabled{background:#fecaca;color:#7f1d1d;}")
        self.btn_stop_alarm.setStyleSheet("QPushButton{background:#b91c1c;color:white;font-size:15px;font-weight:bold;border:none;border-radius:10px;} QPushButton:disabled{background:#fee2e2;color:#991b1b;}")
        self.btn_reset.setStyleSheet("QPushButton{background:white;color:#111827;font-size:14px;border:1px solid #e5e7eb;border-radius:10px;}")
        self.warning_title.setStyleSheet("font-weight:bold;color:#b91c1c;"); self.warning_msg.setStyleSheet("color:#b91c1c;")
        self.how_title.setStyleSheet("font-weight:bold;")
        self.stat_expected_frame.setStyleSheet("background:#f9fafb;border-radius:12px;")
        self.stat_total_frame.setStyleSheet("background:#eff6ff;border-radius:12px;")
        self.stat_passed_frame.setStyleSheet("background:#ecfdf3;border-radius:12px;")
        self.stat_failed_frame.setStyleSheet("background:#fef2f2;border-radius:12px;")
        for w in [self.label_stat_expected,self.label_stat_total,self.label_stat_passed,self.label_stat_failed]: w.setStyleSheet("color:#6b7280;")
        for w in [self.stat_expected_val,self.stat_total_val,self.stat_passed_val,self.stat_failed_val]: w.setStyleSheet("font-size:20px;font-weight:bold;")
        for w in [self.label_stat_expected_unit,self.label_stat_total_unit,self.label_stat_passed_unit,self.label_stat_failed_unit]: w.setStyleSheet("color:#9ca3af;")

        self.combo_expected.clear()
        for n in BATTERY_OPTIONS:
            self.combo_expected.addItem(f"{n} Battery" if n==1 else f"{n} Batteries", n)
        self.combo_expected.setCurrentIndex(2)

        self.proc=None; self.log_thread=None
        self.total_inspected=0; self.total_pass=0; self.total_fail=0
        self.alarm_active=False  # ⭐ 新增：记录是否正在报警
        self.cards_layout=self.findChild(QtWidgets.QGridLayout,"cardsLayout")

        self.btn_start.clicked.connect(self.start_inspection)
        self.btn_stop.clicked.connect(self.stop_inspection)
        self.btn_stop_alarm.clicked.connect(self.stop_alarm)
        self.btn_reset.clicked.connect(self.reset_counters)
        self.combo_expected.currentIndexChanged.connect(self.update_expected_stat)
        self.update_expected_stat()

    # --- helpers ---
    def update_expected_stat(self):
        self.stat_expected_val.setText(str(int(self.combo_expected.currentData())))
    def _update_pass_rate_label(self):
        rate = 0.0 if self.total_inspected==0 else self.total_pass*100.0/self.total_inspected
        self.label_stat_passed_unit.setText(f"{rate:.1f}% rate")
    def _failsafe_gpio_off(self):
        try:
            if SafeBuzzer is not None:
                b=SafeBuzzer(pin=21,active_high=True); b.close()
        except Exception: pass

    # --- buttons ---
    def start_inspection(self):
        if self.proc is not None: return
        expected=int(self.combo_expected.currentData()); self.update_expected_stat()
        cmd=["python3","/home/pi/battery_batch/auto_capture.py","--expected",str(expected),
             "--buzzer_pin","21","--keep_raw","200","--keep_out","500"]
        try:
            self.proc=subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                       stdin=subprocess.PIPE, text=True, bufsize=1)
        except FileNotFoundError:
            QtWidgets.QMessageBox.critical(self,"Error","auto_capture.py not found at /home/pi/battery_batch/")
            self.proc=None; return
        self.log_thread=LogReaderThread(self.proc, expected)
        self.log_thread.newResult.connect(self.on_new_result)
        self.log_thread.processExited.connect(self.on_process_exited)
        self.log_thread.start()
        self.btn_start.setEnabled(False); self.combo_expected.setEnabled(False)
        self.btn_stop.setEnabled(True); self.warning_frame.setVisible(False)
        self.btn_stop_alarm.setEnabled(False)  # 初始不允许

    def stop_inspection(self):
        # ⭐ 规则：报警期间禁止 Stop，必须先 Stop Alarm
        if self.alarm_active:
            QtWidgets.QMessageBox.warning(self, "Alarm is active",
                                          "Please press \"Stop Alarm\" first to silence the buzzer.")
            return
        if self.proc is None: return
        try:
            # 不喂换行，直接优雅退出；auto_capture 有信号/atexit 保证蜂鸣器关闭
            try: self.proc.send_signal(signal.SIGINT)
            except Exception: pass
            try: self.proc.wait(timeout=1.0)
            except subprocess.TimeoutExpired:
                try: self.proc.terminate()
                except Exception: pass
                try: self.proc.wait(timeout=1.0)
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
            self.btn_stop.setEnabled(False); self.btn_stop_alarm.setEnabled(False)
            self.warning_frame.setVisible(False)
            self.alarm_active=False

    def stop_alarm(self):
        if self.proc is None: return
        try:
            # 喂一个回车，解除 auto_capture 的 input() 等待
            self.proc.stdin.write("\n"); self.proc.stdin.flush()
        except Exception: pass
        self._failsafe_gpio_off()
        self.btn_stop_alarm.setEnabled(False)
        self.warning_frame.setVisible(False)
        self.alarm_active=False
        # 报警停了 → 允许 Stop
        if self.proc is not None:
            self.btn_stop.setEnabled(True)

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
        self.total_inspected+=1
        if info.get("ng_count",0)==0: self.total_pass+=1
        else: self.total_fail+=1
        self.stat_total_val.setText(str(self.total_inspected))
        self.stat_passed_val.setText(str(self.total_pass))
        self.stat_failed_val.setText(str(self.total_fail))
        self._update_pass_rate_label()

        if info.get("ng_count",0)>0:
            self.warning_frame.setVisible(True)
            self.btn_stop_alarm.setEnabled(True)
            self.alarm_active=True                 # ⭐ 开启报警标志
            self.btn_stop.setEnabled(False)        # ⭐ 禁用 Stop
        else:
            self.warning_frame.setVisible(False)
            self.btn_stop_alarm.setEnabled(False)
            self.alarm_active=False
            self.btn_stop.setEnabled(True)

        row=(self.total_inspected-1)//3; col=(self.total_inspected-1)%3
        card=ResultCard(info); card.clicked.connect(self.show_detail_dialog)
        self.cards_layout.addWidget(card,row,col); self.empty_label.setVisible(False)

    @QtCore.pyqtSlot(int)
    def on_process_exited(self, code):
        self.proc=None; self.log_thread=None
        self.btn_start.setEnabled(True); self.combo_expected.setEnabled(True)
        self.btn_stop.setEnabled(False); self.btn_stop_alarm.setEnabled(False)
        self.warning_frame.setVisible(False)
        self.alarm_active=False

    def show_detail_dialog(self, info):
        DetailDialog(info, self).exec_()

    def closeEvent(self, e):
        # 如果报警还在，阻止关闭，避免现场误操作
        if self.alarm_active:
            QtWidgets.QMessageBox.warning(self, "Alarm is active",
                                          "Please press \"Stop Alarm\" first, then close.")
            e.ignore(); return
        try: self.stop_inspection()
        except Exception: pass
        e.accept()

def main():
    app = QtWidgets.QApplication(sys.argv)
    win = MainWindow(); win.show()
    sys.exit(app.exec_())

if __name__=="__main__":
    main()
