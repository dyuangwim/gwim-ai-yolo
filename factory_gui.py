import os
import sys
import json
import subprocess
from datetime import datetime

from PyQt5 import QtCore, QtGui, QtWidgets, uic


BATTERY_OPTIONS = [1, 2, 4, 6, 8, 10, 12, 16, 20, 24]


class ResultCard(QtWidgets.QFrame):
    clicked = QtCore.pyqtSignal(dict)

    def __init__(self, info, parent=None):
        super().__init__(parent)
        self.info = info
        self.setObjectName("ResultCard")
        self.setFrameShape(QtWidgets.QFrame.NoFrame)

        border_color = "#22c55e" if info.get("ng_count", 0) == 0 else "#ef4444"
        self.setStyleSheet(f"""
        QFrame#ResultCard {{
            border-radius: 12px;
            border: 2px solid {border_color};
            background: transparent;
        }}
        """)

        root = QtWidgets.QVBoxLayout(self)
        root.setContentsMargins(2, 2, 2, 2)
        root.setSpacing(0)

        # 上：黑色图像区域
        img_frame = QtWidgets.QFrame()
        img_frame.setStyleSheet("background:#020617; border-radius:10px;")
        img_layout = QtWidgets.QVBoxLayout(img_frame)
        img_layout.setContentsMargins(4, 4, 4, 4)

        self.imageLabel = QtWidgets.QLabel()
        self.imageLabel.setAlignment(QtCore.Qt.AlignCenter)
        self.imageLabel.setMinimumSize(220, 120)
        self.imageLabel.setStyleSheet("background:#020617; border-radius:8px; color:#64748b;")
        img_layout.addWidget(self.imageLabel)
        root.addWidget(img_frame, 1)

        # 下：白色信息条
        bottom = QtWidgets.QFrame()
        bottom.setStyleSheet("background:white; border-radius:10px;")
        bottom_layout = QtWidgets.QVBoxLayout(bottom)
        bottom_layout.setContentsMargins(8, 4, 8, 4)
        bottom_layout.setSpacing(2)

        ts = info.get("ts")
        ts_text = ts.strftime("%H:%M:%S") if isinstance(ts, datetime) else str(ts)
        self.timeLabel = QtWidgets.QLabel(ts_text)
        self.timeLabel.setStyleSheet("color:#0f172a; font-weight:bold;")
        bottom_layout.addWidget(self.timeLabel)

        expected = info.get("expected", 0)
        ng = info.get("ng_count", 0)
        pack_count = info.get("pack_count", 0)
        if pack_count:
            text = f"Detected: {expected} batteries/pkg | Packs: {pack_count} | NG: {ng}"
        else:
            text = f"NG packs: {ng}"
        self.summaryLabel = QtWidgets.QLabel(text)
        self.summaryLabel.setStyleSheet("color:#0f172a;")
        bottom_layout.addWidget(self.summaryLabel)

        root.addWidget(bottom, 0)

        # 右上角 Pass / Fail badge
        badge = QtWidgets.QLabel("  Pass  " if info.get("ng_count", 0) == 0 else "  Fail  ", self)
        badge.setAlignment(QtCore.Qt.AlignCenter)
        badge.setStyleSheet(
            "background:%s; color:white; border-radius:10px; "
            "font-weight:bold; padding:2px 6px;"
            % ("#22c55e" if info.get("ng_count", 0) == 0 else "#ef4444")
        )
        badge.move(self.width() - badge.width() - 8, 8)
        badge.raise_()
        self._badge = badge

        if info.get("image_path") and os.path.exists(info["image_path"]):
            pix = QtGui.QPixmap(info["image_path"])
            if not pix.isNull():
                self.set_image(pix)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        # 保证 badge 一直在右上角
        self._badge.move(self.width() - self._badge.width() - 8, 8)

    def set_image(self, pixmap):
        size = self.imageLabel.size()
        if size.width() <= 0 or size.height() <= 0:
            size = QtCore.QSize(220, 120)
        scaled = pixmap.scaled(size, QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation)
        self.imageLabel.setPixmap(scaled)

    def mousePressEvent(self, event):
        if event.button() == QtCore.Qt.LeftButton:
            self.clicked.emit(self.info)
        super().mousePressEvent(event)


class DetailDialog(QtWidgets.QDialog):
    def __init__(self, info, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Inspection Details")
        self.resize(520, 520)
        self.setModal(True)
        self.setStyleSheet("QDialog { background:#f9fafb; }")

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(12)

        header = QtWidgets.QHBoxLayout()
        title = QtWidgets.QLabel("Inspection Details")
        title.setStyleSheet("font-size:18px; font-weight:bold;")
        header.addWidget(title)

        status_label = QtWidgets.QLabel("  Pass  " if info.get("ng_count", 0) == 0 else "  Fail  ")
        status_label.setStyleSheet(
            "background:%s; color:white; border-radius:12px; font-weight:bold; padding:4px 8px;"
            % ("#22c55e" if info.get("ng_count", 0) == 0 else "#ef4444")
        )
        header.addWidget(status_label)
        header.addStretch()
        layout.addLayout(header)

        ts = info.get("ts")
        ts_text = ts.strftime("%Y-%m-%d %H:%M:%S") if isinstance(ts, datetime) else str(ts)
        ts_label = QtWidgets.QLabel(ts_text)
        ts_label.setStyleSheet("color:#6b7280;")
        layout.addWidget(ts_label)

        img_frame = QtWidgets.QFrame()
        img_frame.setStyleSheet("background:white; border-radius:12px;")
        img_layout = QtWidgets.QVBoxLayout(img_frame)
        img_layout.setContentsMargins(8, 8, 8, 8)
        self.imageLabel = QtWidgets.QLabel()
        self.imageLabel.setAlignment(QtCore.Qt.AlignCenter)
        self.imageLabel.setMinimumSize(400, 260)
        self.imageLabel.setStyleSheet("background:#0b1120; border-radius:10px;")
        img_layout.addWidget(self.imageLabel)
        layout.addWidget(img_frame)

        if info.get("image_path") and os.path.exists(info["image_path"]):
            pix = QtGui.QPixmap(info["image_path"])
            if not pix.isNull():
                scaled = pix.scaled(QtCore.QSize(480, 320),
                                    QtCore.Qt.KeepAspectRatio,
                                    QtCore.Qt.SmoothTransformation)
                self.imageLabel.setPixmap(scaled)

        info_frame = QtWidgets.QFrame()
        info_frame.setStyleSheet("background:white; border-radius:12px;")
        info_layout = QtWidgets.QGridLayout(info_frame)
        info_layout.setContentsMargins(12, 8, 12, 8)
        info_layout.setHorizontalSpacing(32)

        expected = info.get("expected", 0)
        ng_count = info.get("ng_count", 0)
        pack_count = info.get("pack_count", 0)

        lbl1 = QtWidgets.QLabel("Expected per pack")
        lbl1.setStyleSheet("color:#6b7280;")
        val1 = QtWidgets.QLabel(f"{expected} batteries")
        val1.setStyleSheet("font-weight:bold;")

        lbl2 = QtWidgets.QLabel("Total packs")
        lbl2.setStyleSheet("color:#6b7280;")
        val2 = QtWidgets.QLabel(str(pack_count))
        val2.setStyleSheet("font-weight:bold;")

        lbl3 = QtWidgets.QLabel("NG packs")
        lbl3.setStyleSheet("color:#6b7280;")
        val3 = QtWidgets.QLabel(str(ng_count))
        val3.setStyleSheet("font-weight:bold; color:%s;" %
                           ("#22c55e" if ng_count == 0 else "#ef4444"))

        info_layout.addWidget(lbl1, 0, 0)
        info_layout.addWidget(val1, 1, 0)
        info_layout.addWidget(lbl2, 0, 1)
        info_layout.addWidget(val2, 1, 1)
        info_layout.addWidget(lbl3, 0, 2)
        info_layout.addWidget(val3, 1, 2)
        layout.addWidget(info_frame)

        issue_frame = QtWidgets.QFrame()
        issue_frame.setStyleSheet(
            "background:%s; border-radius:12px;" %
            ("#dcfce7" if ng_count == 0 else "#fee2e2"))
        issue_layout = QtWidgets.QVBoxLayout(issue_frame)
        issue_layout.setContentsMargins(12, 8, 12, 8)
        issue_title = QtWidgets.QLabel("Status")
        issue_title.setStyleSheet("font-weight:bold;")
        issue_layout.addWidget(issue_title)

        if ng_count == 0:
            msg = "All packs in this image matched the expected quantity."
        else:
            msg = f"Battery count mismatch detected in {ng_count} pack(s). Please check before continuing."
        issue_body = QtWidgets.QLabel(msg)
        issue_body.setWordWrap(True)
        issue_layout.addWidget(issue_body)
        layout.addWidget(issue_frame)

        btn_box = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Close)
        btn_box.rejected.connect(self.reject)
        layout.addWidget(btn_box)


class LogReaderThread(QtCore.QThread):
    newResult = QtCore.pyqtSignal(dict)
    processExited = QtCore.pyqtSignal(int)

    def __init__(self, process: subprocess.Popen, expected_per_pack: int, parent=None):
        super().__init__(parent)
        self._process = process
        self._expected = expected_per_pack

    def run(self):
        proc = self._process
        current_image = None
        json_path = None

        while True:
            line = proc.stdout.readline()
            if not line:
                break
            line = line.strip()

            if line.startswith("Image: "):
                current_image = line.split("Image:", 1)[1].strip()
            elif line.startswith("JSON:"):
                json_path = line.split("JSON:", 1)[1].strip()
                info = self._build_info(current_image, json_path)
                if info:
                    self.newResult.emit(info)
                current_image = None
                json_path = None

        proc.wait()
        self.processExited.emit(proc.returncode)

    def _build_info(self, image_path, json_path):
        if not image_path:
            return None
        info = {
            "image_path": image_path,
            "expected": self._expected,
            "ts": datetime.now(),
            "ng_count": 0,
            "pack_count": 0,
            "packs": [],
            "json_path": json_path,
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
                print("Failed to parse JSON:", e, file=sys.stderr)
        return info


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()

        ui_path = os.path.join(os.path.dirname(__file__), "batch_inspector.ui")
        uic.loadUi(ui_path, self)

        # 全局样式
        self.setStyleSheet("""
        QMainWindow { background:#f1f5f9; }
        QLabel { font-size:14px; }
        QComboBox, QSpinBox { font-size:14px; }
        QFrame#controlCard { background:white; border-radius:16px; }
        QFrame#stats_card { background:white; border-radius:16px; }
        QFrame#results_card { background:white; border-radius:16px; }
        QFrame#how_frame { background:#eff6ff; border-radius:10px; }
        QFrame#warning_frame { background:#fee2e2; border-radius:10px; }
        QLabel#label_cp_title { font-size:16px; font-weight:bold; }
        QLabel#title_main { font-size:20px; font-weight:bold; }
        QLabel#title_sub { color:#6b7280; }
        QLabel#empty_label { color:#9ca3af; }
        """)

        # 控件样式
        self.combo_expected.setStyleSheet("""
        QComboBox {
            padding:8px 10px;
            border-radius:8px;
            border:1px solid #cbd5e1;
            background:white;
        }""")
        self.btn_start.setStyleSheet("""
        QPushButton {
            background:#16a34a; color:white; font-size:16px;
            font-weight:bold; border:none; border-radius:10px;
        }
        QPushButton:disabled {
            background:#9ca3af;
        }""")
        self.btn_stop.setStyleSheet("""
        QPushButton {
            background:#ef4444; color:white; font-size:15px;
            font-weight:bold; border:none; border-radius:10px;
        }
        QPushButton:disabled {
            background:#fecaca; color:#7f1d1d;
        }""")
        self.btn_stop_alarm.setStyleSheet("""
        QPushButton {
            background:#b91c1c; color:white; font-size:15px;
            font-weight:bold; border:none; border-radius:10px;
        }
        QPushButton:disabled {
            background:#fee2e2; color:#991b1b;
        }""")
        self.btn_reset.setStyleSheet("""
        QPushButton {
            background:white; color:#111827; font-size:14px;
            border:1px solid #e5e7eb; border-radius:10px;
        }""")
        self.warning_title.setStyleSheet("font-weight:bold; color:#b91c1c;")
        self.warning_msg.setStyleSheet("color:#b91c1c;")
        self.how_title.setStyleSheet("font-weight:bold;")

        # stats 卡片颜色和字体
        self.stat_expected_frame.setStyleSheet("background:#f9fafb; border-radius:12px;")
        self.stat_total_frame.setStyleSheet("background:#eff6ff; border-radius:12px;")
        self.stat_passed_frame.setStyleSheet("background:#ecfdf3; border-radius:12px;")
        self.stat_failed_frame.setStyleSheet("background:#fef2f2; border-radius:12px;")

        for lbl in [
            self.label_stat_expected,
            self.label_stat_total,
            self.label_stat_passed,
            self.label_stat_failed,
        ]:
            lbl.setStyleSheet("color:#6b7280;")
        for lbl in [
            self.stat_expected_val,
            self.stat_total_val,
            self.stat_passed_val,
            self.stat_failed_val,
        ]:
            lbl.setStyleSheet("font-size:20px; font-weight:bold;")
        for lbl in [
            self.label_stat_expected_unit,
            self.label_stat_total_unit,
            self.label_stat_passed_unit,
            self.label_stat_failed_unit,
        ]:
            lbl.setStyleSheet("color:#9ca3af;")

        # combobox 选项
        self.combo_expected.clear()
        for n in BATTERY_OPTIONS:
            txt = f"{n} Battery" if n == 1 else f"{n} Batteries"
            self.combo_expected.addItem(txt, n)
        self.combo_expected.setCurrentIndex(2)  # 默认 4

        # 统计计数
        self.proc = None
        self.log_thread = None
        self.total_inspected = 0
        self.total_pass = 0
        self.total_fail = 0

        self.cards_layout = self.findChild(QtWidgets.QGridLayout, "cardsLayout")

        # 信号
        self.btn_start.clicked.connect(self.start_inspection)
        self.btn_stop.clicked.connect(self.stop_inspection)
        self.btn_stop_alarm.clicked.connect(self.stop_alarm)
        self.btn_reset.clicked.connect(self.reset_counters)
        self.combo_expected.currentIndexChanged.connect(self.update_expected_stat)

        self.update_expected_stat()

    # ---- 控制逻辑 ----

    def update_expected_stat(self):
        expected = int(self.combo_expected.currentData())
        self.stat_expected_val.setText(str(expected))

    def _update_pass_rate_label(self):
        if self.total_inspected == 0:
            rate = 0.0
        else:
            rate = self.total_pass * 100.0 / self.total_inspected
        self.label_stat_passed_unit.setText(f"{rate:.1f}% rate")

    def start_inspection(self):
        if self.proc is not None:
            return
        expected = int(self.combo_expected.currentData())
        self.update_expected_stat()

        cmd = [
            "python3",
            "/home/pi/battery_batch/auto_capture.py",
            "--expected",
            str(expected),
            "--buzzer_pin",
            "21",
            "--keep_raw",
            "200",
            "--keep_out",
            "500",
        ]
        try:
            self.proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                stdin=subprocess.PIPE,
                text=True,
                bufsize=1,
            )
        except FileNotFoundError:
            QtWidgets.QMessageBox.critical(
                self, "Error",
                "auto_capture.py not found at /home/pi/battery_batch/")
            self.proc = None
            return

        self.log_thread = LogReaderThread(self.proc, expected)
        self.log_thread.newResult.connect(self.on_new_result)
        self.log_thread.processExited.connect(self.on_process_exited)
        self.log_thread.start()

        self.btn_start.setEnabled(False)
        self.combo_expected.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.warning_frame.setVisible(False)

    def stop_inspection(self):
        if self.proc is None:
            return
        try:
            self.proc.terminate()
            self.proc.stdin.close()
        except Exception:
            pass
        self.proc = None

    def stop_alarm(self):
        if self.proc is None:
            return
        try:
            self.proc.stdin.write("\n")
            self.proc.stdin.flush()
        except Exception:
            pass
        self.btn_stop_alarm.setEnabled(False)
        self.warning_frame.setVisible(False)

    def reset_counters(self):
        self.total_inspected = 0
        self.total_pass = 0
        self.total_fail = 0
        self.stat_total_val.setText("0")
        self.stat_passed_val.setText("0")
        self.stat_failed_val.setText("0")
        self._update_pass_rate_label()
        # 清空卡片
        while self.cards_layout.count():
            item = self.cards_layout.takeAt(0)
            w = item.widget()
            if w:
                w.deleteLater()
        self.empty_label.setVisible(True)

    @QtCore.pyqtSlot(dict)
    def on_new_result(self, info):
        self.total_inspected += 1
        if info.get("ng_count", 0) == 0:
            self.total_pass += 1
        else:
            self.total_fail += 1

        self.stat_total_val.setText(str(self.total_inspected))
        self.stat_passed_val.setText(str(self.total_pass))
        self.stat_failed_val.setText(str(self.total_fail))
        self._update_pass_rate_label()

        if info.get("ng_count", 0) > 0:
            self.warning_frame.setVisible(True)
            self.btn_stop_alarm.setEnabled(True)
        else:
            self.warning_frame.setVisible(False)

        row = (self.total_inspected - 1) // 3
        col = (self.total_inspected - 1) % 3

        card = ResultCard(info)
        card.clicked.connect(self.show_detail_dialog)
        self.cards_layout.addWidget(card, row, col)
        self.empty_label.setVisible(False)

    @QtCore.pyqtSlot(int)
    def on_process_exited(self, code):
        self.proc = None
        self.log_thread = None
        self.btn_start.setEnabled(True)
        self.combo_expected.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.btn_stop_alarm.setEnabled(False)

    def show_detail_dialog(self, info):
        dlg = DetailDialog(info, self)
        dlg.exec_()

    def closeEvent(self, event):
        if self.proc is not None:
            try:
                self.proc.terminate()
                self.proc.stdin.close()
            except Exception:
                pass
        event.accept()


def main():
    app = QtWidgets.QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
