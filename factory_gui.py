import os, sys, json, signal, subprocess, time
from datetime import datetime
from PyQt5 import QtCore, QtGui, QtWidgets, uic

# 仅用于"Stop/关闭窗口时"的最终保险；Stop Alarm 不触GPIO
try:
    from utils_hw import Buzzer as SafeBuzzer
except Exception:
    SafeBuzzer = None

BATTERY_OPTIONS = [1, 2, 4, 6, 8, 10, 12, 16, 20, 24]

class ResultCard(QtWidgets.QFrame):
    clicked = QtCore.pyqtSignal(dict)
    def __init__(self, info, parent=None):
        super().__init__(parent)
        self.info = info
        self.setObjectName("ResultCard")
        self.setFrameShape(QtWidgets.QFrame.NoFrame)
        border_color = "#22c55e" if info.get("ng_count",0)==0 else "#ef4444"
        self.setStyleSheet(f"QFrame#ResultCard{{border-radius:12px;border:2px solid {border_color};background:transparent;}}")
        
        root = QtWidgets.QVBoxLayout(self)
        root.setContentsMargins(2,2,2,2)
        root.setSpacing(0)
        
        imgf = QtWidgets.QFrame()
        imgf.setStyleSheet("background:#020617;border-radius:10px;")
        il = QtWidgets.QVBoxLayout(imgf)
        il.setContentsMargins(4,4,4,4)
        
        self.imageLabel = QtWidgets.QLabel(alignment=QtCore.Qt.AlignCenter)
        self.imageLabel.setMinimumSize(220,120)
        self.imageLabel.setStyleSheet("background:#020617;border-radius:8px;color:#64748b;")
        il.addWidget(self.imageLabel)
        root.addWidget(imgf,1)
        
        bottom = QtWidgets.QFrame()
        bottom.setStyleSheet("background:white;border-radius:10px;")
        bl = QtWidgets.QVBoxLayout(bottom)
        bl.setContentsMargins(8,4,8,4)
        bl.setSpacing(2)
        
        ts = info.get("ts")
        ts_text = ts.strftime("%H:%M:%S") if isinstance(ts, datetime) else str(ts)
        t = QtWidgets.QLabel(ts_text)
        t.setStyleSheet("color:#0f172a;font-weight:bold;")
        bl.addWidget(t)
        
        expected, ng, pc = info.get("expected",0), info.get("ng_count",0), info.get("pack_count",0)
        s = f"Detected: {expected} batteries/pkg | Packs: {pc} | NG: {ng}" if pc else f"NG packs: {ng}"
        x = QtWidgets.QLabel(s)
        x.setStyleSheet("color:#0f172a;")
        bl.addWidget(x)
        root.addWidget(bottom,0)
        
        badge = QtWidgets.QLabel("  Pass  " if ng==0 else "  Fail  ", self)
        badge.setStyleSheet("background:%s;color:white;border-radius:10px;font-weight:bold;padding:2px 6px;" % ("#22c55e" if ng==0 else "#ef4444"))
        badge.move(self.width()-badge.width()-8,8)
        badge.raise_()
        self._badge = badge
        
        if info.get("image_path") and os.path.exists(info["image_path"]):
            p = QtGui.QPixmap(info["image_path"])
            if not p.isNull():
                self.set_image(p)
                
    def resizeEvent(self, e):
        super().resizeEvent(e)
        self._badge.move(self.width()-self._badge.width()-8,8)
        
    def set_image(self, p):
        size = self.imageLabel.size() or QtCore.QSize(220,120)
        self.imageLabel.setPixmap(p.scaled(size, QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation))
        
    def mousePressEvent(self, e):
        if e.button() == QtCore.Qt.LeftButton:
            self.clicked.emit(self.info)
        super().mousePressEvent(e)

class DetailDialog(QtWidgets.QDialog):
    def __init__(self, info, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Inspection Details")
        self.resize(520,520)
        self.setModal(True)
        self.setStyleSheet("QDialog{background:#f9fafb;}")
        
        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(16,16,16,16)
        lay.setSpacing(12)
        
        hdr = QtWidgets.QHBoxLayout()
        t = QtWidgets.QLabel("Inspection Details")
        t.setStyleSheet("font-size:18px;font-weight:bold;")
        hdr.addWidget(t)
        
        status = QtWidgets.QLabel("  Pass  " if info.get("ng_count",0)==0 else "  Fail  ")
        status.setStyleSheet("background:%s;color:white;border-radius:12px;font-weight:bold;padding:4px 8px;" % ("#22c55e" if info.get("ng_count",0)==0 else "#ef4444"))
        hdr.addWidget(status)
        hdr.addStretch()
        lay.addLayout(hdr)
        
        ts = info.get("ts")
        ts_label = QtWidgets.QLabel(ts.strftime("%Y-%m-%d %H:%M:%S") if isinstance(ts, datetime) else str(ts))
        ts_label.setStyleSheet("color:#6b7280;")
        lay.addWidget(ts_label)
        
        imgf = QtWidgets.QFrame()
        imgf.setStyleSheet("background:white;border-radius:12px;")
        il = QtWidgets.QVBoxLayout(imgf)
        il.setContentsMargins(8,8,8,8)
        
        img = QtWidgets.QLabel(alignment=QtCore.Qt.AlignCenter)
        img.setMinimumSize(400,260)
        img.setStyleSheet("background:#0b1120;border-radius:10px;")
        il.addWidget(img)
        lay.addWidget(imgf)
        
        if info.get("image_path") and os.path.exists(info["image_path"]):
            p = QtGui.QPixmap(info["image_path"])
            if not p.isNull():
                img.setPixmap(p.scaled(QtCore.QSize(480,320), QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation))
        
        inf = QtWidgets.QFrame()
        inf.setStyleSheet("background:white;border-radius:12px;")
        grid = QtWidgets.QGridLayout(inf)
        grid.setContentsMargins(12,8,12,8)
        grid.setHorizontalSpacing(32)
        
        exp, ng, pc = info.get("expected",0), info.get("ng_count",0), info.get("pack_count",0)
        labels = [
            ("Expected per pack", f"{exp} batteries"),
            ("Total packs", str(pc)),
            ("NG packs", str(ng))
        ]
        
        for i,(k,v) in enumerate(labels):
            a = QtWidgets.QLabel(k)
            a.setStyleSheet("color:#6b7280;")
            b = QtWidgets.QLabel(v)
            b.setStyleSheet("font-weight:bold;" + ("color:#22c55e;" if k=="NG packs" and ng==0 else ""))
            grid.addWidget(a,0,i)
            grid.addWidget(b,1,i)
        lay.addWidget(inf)
        
        box = QtWidgets.QFrame()
        box.setStyleSheet("background:%s;border-radius:12px;" % ("#dcfce7" if ng==0 else "#fee2e2"))
        bl = QtWidgets.QVBoxLayout(box)
        bl.setContentsMargins(12,8,12,8)
        
        title = QtWidgets.QLabel("Status")
        title.setStyleSheet("font-weight:bold;")
        bl.addWidget(title)
        
        msg = "All packs in this image matched the expected quantity." if ng==0 else f"Battery count mismatch detected in {ng} pack(s). Please check before continuing."
        body = QtWidgets.QLabel(msg)
        body.setWordWrap(True)
        bl.addWidget(body)
        lay.addWidget(box)
        
        btn = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Close)
        btn.rejected.connect(self.reject)
        lay.addWidget(btn)

class LogReaderThread(QtCore.QThread):
    newResult = QtCore.pyqtSignal(dict)
    processExited = QtCore.pyqtSignal(int)
    logMessage = QtCore.pyqtSignal(str)  # 新增：用于调试
    
    def __init__(self, process: subprocess.Popen, expected_per_pack: int, parent=None):
        super().__init__(parent)
        self._process = process
        self._expected = expected_per_pack
        self._stopped = False
        
    def stop(self):
        """停止读取线程"""
        self._stopped = True
        
    def run(self):
        p = self._process
        current_img = None
        current_json = None
        
        try:
            while not self._stopped:
                # 非阻塞读取，带超时
                line = p.stdout.readline()
                
                if not line:  # EOF
                    self.logMessage.emit("[LogReader] EOF reached")
                    break
                    
                s = line.strip()
                if s:  # 只处理非空行
                    self.logMessage.emit(f"[LogReader] {s}")
                
                # 检测到新图片
                if s.startswith("Image: "):
                    current_img = s.split("Image:", 1)[1].strip()
                    current_json = None  # 重置 JSON 路径
                    self.logMessage.emit(f"[LogReader] Captured image: {current_img}")
                    
                # 检测到 JSON（完整检测周期）
                elif s.startswith("JSON:"):
                    current_json = s.split("JSON:", 1)[1].strip()
                    self.logMessage.emit(f"[LogReader] Captured JSON: {current_json}")
                    
                    # 只有当 Image 和 JSON 都存在时才发射信号
                    if current_img and current_json:
                        info = self._build_info(current_img, current_json)
                        if info:
                            self.logMessage.emit(f"[LogReader] Emitting result: NG={info.get('ng_count',0)}")
                            self.newResult.emit(info)
                        
                        # 重置状态，防止重复
                        current_img = None
                        current_json = None
                        
        except Exception as e:
            self.logMessage.emit(f"[LogReader] Error: {e}")
            print(f"LogReaderThread error: {e}", file=sys.stderr)
        finally:
            # 等待进程结束
            try:
                p.wait(timeout=2)
            except subprocess.TimeoutExpired:
                pass
            self.logMessage.emit(f"[LogReader] Process exited with code: {p.returncode}")
            self.processExited.emit(p.returncode if p.returncode is not None else -1)
    
    def _build_info(self, image_path, json_path):
        """构建结果信息字典"""
        if not image_path:
            return None
            
        info = {
            "image_path": image_path,
            "expected": self._expected,
            "ts": datetime.now(),
            "ng_count": 0,
            "pack_count": 0,
            "packs": [],
            "json_path": json_path
        }
        
        if json_path and os.path.exists(json_path):
            try:
                with open(json_path, "r") as f:
                    data = json.load(f)
                packs = data.get("packs", [])
                info["packs"] = packs
                info["pack_count"] = len(packs)
                info["ng_count"] = len([p for p in packs if not p.get("ok", True)])
            except Exception as e:
                print(f"Failed to parse JSON: {e}", file=sys.stderr)
                
        return info

class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        uic.loadUi(os.path.join(os.path.dirname(__file__),"batch_inspector.ui"), self)
        
        # 样式表
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
        
        self.warning_title.setStyleSheet("font-weight:bold;color:#b91c1c;")
        self.warning_msg.setStyleSheet("color:#b91c1c;")
        self.how_title.setStyleSheet("font-weight:bold;")
        
        self.stat_expected_frame.setStyleSheet("background:#f9fafb;border-radius:12px;")
        self.stat_total_frame.setStyleSheet("background:#eff6ff;border-radius:12px;")
        self.stat_passed_frame.setStyleSheet("background:#ecfdf3;border-radius:12px;")
        self.stat_failed_frame.setStyleSheet("background:#fef2f2;border-radius:12px;")
        
        for x in [self.label_stat_expected,self.label_stat_total,self.label_stat_passed,self.label_stat_failed]:
            x.setStyleSheet("color:#6b7280;")
        for x in [self.stat_expected_val,self.stat_total_val,self.stat_passed_val,self.stat_failed_val]:
            x.setStyleSheet("font-size:20px;font-weight:bold;")
        for x in [self.label_stat_expected_unit,self.label_stat_total_unit,self.label_stat_passed_unit,self.label_stat_failed_unit]:
            x.setStyleSheet("color:#9ca3af;")

        # 初始化选项
        self.combo_expected.clear()
        for n in BATTERY_OPTIONS:
            self.combo_expected.addItem(f"{n} Battery" if n==1 else f"{n} Batteries", n)
        self.combo_expected.setCurrentIndex(2)

        # 状态变量
        self.proc = None
        self.log_thread = None
        self.total_inspected = 0
        self.total_pass = 0
        self.total_fail = 0
        self.cards_layout = self.findChild(QtWidgets.QGridLayout,"cardsLayout")
        self.alarm_active = False

        # 连接信号
        self.btn_start.clicked.connect(self.start_inspection)
        self.btn_stop.clicked.connect(self.stop_inspection)
        self.btn_stop_alarm.clicked.connect(self.stop_alarm)
        self.btn_reset.clicked.connect(self.reset_counters)
        self.combo_expected.currentIndexChanged.connect(self.update_expected_stat)

        self.update_expected_stat()
        self.btn_stop_alarm.setEnabled(False)

    def update_expected_stat(self):
        self.stat_expected_val.setText(str(int(self.combo_expected.currentData())))

    def _update_pass_rate_label(self):
        rate = 0.0 if self.total_inspected == 0 else self.total_pass * 100.0 / self.total_inspected
        self.label_stat_passed_unit.setText(f"{rate:.1f}% rate")

    def start_inspection(self):
        if self.proc is not None:
            print("[GUI] Process already running", file=sys.stderr)
            return
            
        expected = int(self.combo_expected.currentData())
        self.update_expected_stat()
        
        print(f"[GUI] Starting inspection with expected={expected}", file=sys.stderr)
        
        cmd = [
            "python3", "/home/pi/battery_batch/auto_capture.py",
            "--expected", str(expected),
            "--buzzer_pin", "21",
            "--keep_raw", "200",
            "--keep_out", "500"
        ]
        
        try:
            # 创建新进程，确保 stdin 是可用的
            self.proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                stdin=subprocess.PIPE,
                text=True,
                bufsize=1,
                universal_newlines=True
            )
            print(f"[GUI] Process started with PID: {self.proc.pid}", file=sys.stderr)
            
        except FileNotFoundError:
            QtWidgets.QMessageBox.critical(
                self, "Error",
                "auto_capture.py not found at /home/pi/battery_batch/"
            )
            self.proc = None
            return
        except Exception as e:
            QtWidgets.QMessageBox.critical(
                self, "Error",
                f"Failed to start process: {e}"
            )
            self.proc = None
            return
            
        # 启动日志读取线程
        self.log_thread = LogReaderThread(self.proc, expected)
        self.log_thread.newResult.connect(self.on_new_result)
        self.log_thread.processExited.connect(self.on_process_exited)
        self.log_thread.logMessage.connect(self.on_log_message)
        self.log_thread.start()
        
        # 更新 UI 状态
        self.btn_start.setEnabled(False)
        self.combo_expected.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.warning_frame.setVisible(False)
        
        # 确保开始时不在报警状态
        self.alarm_active = False
        self.btn_stop_alarm.setEnabled(False)
        
        print("[GUI] Inspection started successfully", file=sys.stderr)

    def _failsafe_gpio_off(self):
        """只在 Stop/关闭窗口调用；Stop Alarm 不触GPIO，避免与子进程竞态"""
        try:
            if SafeBuzzer is not None:
                b = SafeBuzzer(pin=21, active_high=True)
                b.off()
                b.close()
                print("[GUI] Failsafe GPIO off executed", file=sys.stderr)
        except Exception as e:
            print(f"[GUI] Failsafe GPIO off error: {e}", file=sys.stderr)

    def stop_inspection(self):
        """停止检测流程"""
        print("[GUI] Stop inspection requested", file=sys.stderr)
        
        # 禁止在报警期间按 Stop
        if self.alarm_active:
            QtWidgets.QMessageBox.warning(
                self, "Alarm active",
                "请先点击『Stop Alarm』静音报警，再停止流程。"
            )
            return

        if self.proc is None:
            print("[GUI] No process to stop", file=sys.stderr)
            return
            
        try:
            print(f"[GUI] Stopping process PID: {self.proc.pid}", file=sys.stderr)
            
            # 停止日志读取线程
            if self.log_thread:
                self.log_thread.stop()
                
            # 尝试优雅地终止子进程
            try:
                if self.proc.stdin and not self.proc.stdin.closed:
                    print("[GUI] Sending newline to stdin", file=sys.stderr)
                    self.proc.stdin.write("\n")
                    self.proc.stdin.flush()
                    time.sleep(0.2)  # 给子进程一点时间处理
            except Exception as e:
                print(f"[GUI] Failed to write to stdin: {e}", file=sys.stderr)
                
            # 发送 SIGINT
            try:
                self.proc.send_signal(signal.SIGINT)
                print("[GUI] SIGINT sent", file=sys.stderr)
            except Exception as e:
                print(f"[GUI] Failed to send SIGINT: {e}", file=sys.stderr)
                
            # 等待进程结束
            try:
                self.proc.wait(timeout=2.0)
                print("[GUI] Process ended gracefully", file=sys.stderr)
            except subprocess.TimeoutExpired:
                print("[GUI] Process timeout, terminating...", file=sys.stderr)
                try:
                    self.proc.terminate()
                    self.proc.wait(timeout=1.0)
                except subprocess.TimeoutExpired:
                    print("[GUI] Process still alive, killing...", file=sys.stderr)
                    try:
                        self.proc.kill()
                    except Exception:
                        pass
                        
            # 确保 GPIO 关闭
            self._failsafe_gpio_off()
            
        finally:
            # 清理资源
            try:
                if self.proc and self.proc.stdin and not self.proc.stdin.closed:
                    self.proc.stdin.close()
            except Exception as e:
                print(f"[GUI] Failed to close stdin: {e}", file=sys.stderr)
                
            # 等待日志线程结束
            if self.log_thread:
                if not self.log_thread.wait(2000):  # 等待最多 2 秒
                    print("[GUI] Log thread didn't finish in time", file=sys.stderr)
                self.log_thread = None
                
            self.proc = None
            
            # 更新 UI
            self.btn_start.setEnabled(True)
            self.combo_expected.setEnabled(True)
            self.btn_stop.setEnabled(False)
            self.btn_stop_alarm.setEnabled(False)
            self.warning_frame.setVisible(False)
            self.alarm_active = False
            
            print("[GUI] Stop inspection complete", file=sys.stderr)

    def stop_alarm(self):
        """仅通知子进程停报警（stdin 回车），不触GPIO；恢复 Stop 的可用性"""
        print("[GUI] Stop alarm requested", file=sys.stderr)
        
        if self.proc is None:
            print("[GUI] No process running", file=sys.stderr)
            return
            
        try:
            if self.proc.stdin and not self.proc.stdin.closed:
                print("[GUI] Sending newline to stop alarm", file=sys.stderr)
                self.proc.stdin.write("\n")
                self.proc.stdin.flush()
                print("[GUI] Stop alarm signal sent", file=sys.stderr)
        except Exception as e:
            print(f"[GUI] Failed to send stop alarm signal: {e}", file=sys.stderr)
            QtWidgets.QMessageBox.warning(
                self, "Warning",
                f"Failed to stop alarm: {e}\n\nYou may need to restart the inspection."
            )
            
        # 更新状态
        self.alarm_active = False
        self.btn_stop_alarm.setEnabled(False)
        self.warning_frame.setVisible(False)
        self.btn_stop.setEnabled(True)

    def reset_counters(self):
        """重置计数器和结果卡片"""
        self.total_inspected = 0
        self.total_pass = 0
        self.total_fail = 0
        
        self.stat_total_val.setText("0")
        self.stat_passed_val.setText("0")
        self.stat_failed_val.setText("0")
        self._update_pass_rate_label()
        
        # 清空所有卡片
        while self.cards_layout.count():
            item = self.cards_layout.takeAt(0)
            widget = item.widget()
            if widget:
                widget.deleteLater()
                
        self.empty_label.setVisible(True)

    @QtCore.pyqtSlot(dict)
    def on_new_result(self, info):
        """处理新的检测结果"""
        print(f"[GUI] New result received: NG={info.get('ng_count',0)}", file=sys.stderr)
        
        self.total_inspected += 1
        
        if info.get("ng_count", 0) == 0:
            self.total_pass += 1
        else:
            self.total_fail += 1
            
        # 更新统计数据
        self.stat_total_val.setText(str(self.total_inspected))
        self.stat_passed_val.setText(str(self.total_pass))
        self.stat_failed_val.setText(str(self.total_fail))
        self._update_pass_rate_label()

        # 处理报警状态
        if info.get("ng_count", 0) > 0:
            # NG 检测到：进入报警状态
            print("[GUI] NG detected, activating alarm UI", file=sys.stderr)
            self.warning_frame.setVisible(True)
            self.alarm_active = True
            self.btn_stop_alarm.setEnabled(True)
            self.btn_stop.setEnabled(False)  # 报警时禁用 Stop
        else:
            # Pass：确保不在报警状态
            print("[GUI] Pass detected, no alarm", file=sys.stderr)
            self.warning_frame.setVisible(False)
            self.alarm_active = False
            self.btn_stop_alarm.setEnabled(False)
            self.btn_stop.setEnabled(True if self.proc is not None else False)

        # 添加结果卡片
        row = (self.total_inspected - 1) // 3
        col = (self.total_inspected - 1) % 3
        card = ResultCard(info)
        card.clicked.connect(self.show_detail_dialog)
        self.cards_layout.addWidget(card, row, col)
        self.empty_label.setVisible(False)

    @QtCore.pyqtSlot(int)
    def on_process_exited(self, code):
        """子进程退出处理"""
        print(f"[GUI] Process exited with code: {code}", file=sys.stderr)
        
        self.proc = None
        self.log_thread = None
        
        # 更新 UI
        self.btn_start.setEnabled(True)
        self.combo_expected.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.btn_stop_alarm.setEnabled(False)
        self.warning_frame.setVisible(False)
        self.alarm_active = False

    @QtCore.pyqtSlot(str)
    def on_log_message(self, msg):
        """处理日志消息（调试用）"""
        # 可以选择性地显示在控制台
        if "[Alarm]" in msg or "[WaitAlarm]" in msg or "NG" in msg:
            print(msg, file=sys.stderr)

    def show_detail_dialog(self, info):
        """显示详细信息对话框"""
        DetailDialog(info, self).exec_()

    def closeEvent(self, e):
        """窗口关闭事件"""
        print("[GUI] Close event triggered", file=sys.stderr)
        
        # 如果在报警，先停止
        if self.alarm_active:
            print("[GUI] Stopping alarm before close", file=sys.stderr)
            self.stop_alarm()
            time.sleep(0.3)  # 给子进程一点时间响应
            
        # 停止检测
        try:
            self.stop_inspection()
        except Exception as e:
            print(f"[GUI] Error during close: {e}", file=sys.stderr)
            
        e.accept()

def main():
    app = QtWidgets.QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
