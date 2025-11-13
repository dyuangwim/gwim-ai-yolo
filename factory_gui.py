import os
import sys
import json
import subprocess
import threading
from datetime import datetime

from PyQt5 import QtCore, QtGui, QtWidgets


# ----------------------
# Helpers
# ----------------------

BATTERY_OPTIONS = [1, 2, 4, 6, 8, 10, 12, 16, 20, 24]


class ResultCard(QtWidgets.QFrame):
    clicked = QtCore.pyqtSignal(dict)  # emit info dict when clicked

    def __init__(self, info, parent=None):
        super().__init__(parent)
        self.info = info  # dict: {image_path, ts, expected, ng_count, pack_count}
        self.setObjectName("ResultCard")
        self.setFrameShape(QtWidgets.QFrame.Box)
        self.setLineWidth(0)
        self.setStyleSheet("""
        QFrame#ResultCard {
            border-radius: 12px;
            border: 2px solid %s;
            background: #000000;
        }
        """ % ("#22c55e" if info.get("ng_count", 0) == 0 else "#ef4444"))

        vbox = QtWidgets.QVBoxLayout(self)
        vbox.setContentsMargins(2, 2, 2, 2)
        vbox.setSpacing(0)

        self.imageLabel = QtWidgets.QLabel()
        self.imageLabel.setMinimumSize(240, 135)
        self.imageLabel.setAlignment(QtCore.Qt.AlignCenter)
        self.imageLabel.setStyleSheet("background:#020617; color:#64748b; border-radius:10px;")
        vbox.addWidget(self.imageLabel, 1)

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
            summary = f"Packs: {pack_count}  |  NG: {ng}"
        else:
            summary = f"NG packs: {ng}"
        self.summaryLabel = QtWidgets.QLabel(summary)
        self.summaryLabel.setStyleSheet("color:#0f172a;")
        bottom_layout.addWidget(self.summaryLabel)

        vbox.addWidget(bottom, 0)

        badge = QtWidgets.QLabel("  Pass  " if info.get("ng_count", 0) == 0 else "  Fail  ")
        badge.setAlignment(QtCore.Qt.AlignCenter)
        badge.setStyleSheet(
            "background:%s; color:white; border-radius:10px; font-weight:bold; padding:2px 6px;"
            % ("#22c55e" if info.get("ng_count", 0) == 0 else "#ef4444")
        )
        badge.move(8, 8)
        badge.setParent(self)

        self._badge = badge

        if info.get("image_path") and os.path.exists(info["image_path"]):
            pix = QtGui.QPixmap(info["image_path"])
            if not pix.isNull():
                self.set_image(pix)

    def set_image(self, pixmap):
        label_size = self.imageLabel.size()
        if label_size.width() <= 0 or label_size.height() <= 0:
            label_size = QtCore.QSize(240, 135)
        scaled = pixmap.scaled(label_size, QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation)
        self.imageLabel.setPixmap(scaled)

    def mousePressEvent(self, event):
        if event.button() == QtCore.Qt.LeftButton:
            self.clicked.emit(self.info)
        super().mousePressEvent(event)


class DetailDialog(QtWidgets.QDialog):
    def __init__(self, info, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Inspection Details")
        self.setModal(True)
        self.resize(520, 520)
        self.setStyleSheet("""
        QDialog {
            background:#f9fafb;
        }
        """)
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(12)

        header = QtWidgets.QHBoxLayout()
        title = QtWidgets.QLabel("Inspection Details")
        title.setStyleSheet("font-size:18px; font-weight:bold;")
        header.addWidget(title)

        status_label = QtWidgets.QLabel("  Pass  " if info.get("ng_count", 0) == 0 else "  Fail  ")
        status_label.setAlignment(QtCore.Qt.AlignCenter)
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

        # image
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
                scaled = pix.scaled(QtCore.QSize(480, 320), QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation)
                self.imageLabel.setPixmap(scaled)

        # expected vs detected
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
        val3.setStyleSheet("font-weight:bold; color:%s;" % ("#22c55e" if ng_count == 0 else "#ef4444"))

        info_layout.addWidget(lbl1, 0, 0)
        info_layout.addWidget(val1, 1, 0)
        info_layout.addWidget(lbl2, 0, 1)
        info_layout.addWidget(val2, 1, 1)
        info_layout.addWidget(lbl3, 0, 2)
        info_layout.addWidget(val3, 1, 2)

        layout.addWidget(info_frame)

        # issue description
        issue_frame = QtWidgets.QFrame()
        issue_frame.setStyleSheet(
            "background:%s; border-radius:12px;"
            % ("#dcfce7" if ng_count == 0 else "#fee2e2")
        )
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
            # print("LOG:", line)  # debug

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
        self.setWindowTitle("Battery Package Inspection")
        self.resize(1366, 768)
        self.setStyleSheet("""
        QMainWindow {
            background:#f1f5f9;
        }
        QLabel {
            font-size:14px;
        }
        QComboBox, QSpinBox {
            font-size:14px;
        }
        QPlainTextEdit {
            background:white;
            border-radius:8px;
        }
        """)

        self.proc = None
        self.log_thread = None
        self.total_inspected = 0
        self.total_pass = 0
        self.total_fail = 0

        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        main_layout = QtWidgets.QHBoxLayout(central)
        main_layout.setContentsMargins(16, 16, 16, 16)
        main_layout.setSpacing(16)

        # Left control panel
        control_card = QtWidgets.QFrame()
        control_card.setStyleSheet("background:white; border-radius:16px;")
        control_layout = QtWidgets.QVBoxLayout(control_card)
        control_layout.setContentsMargins(20, 20, 20, 20)
        control_layout.setSpacing(16)

        title_row = QtWidgets.QHBoxLayout()
        icon_lbl = QtWidgets.QLabel("🖥️")
        icon_lbl.setFixedWidth(24)
        title_row.addWidget(icon_lbl)
        cp_title = QtWidgets.QLabel("Control Panel")
        cp_title.setStyleSheet("font-size:16px; font-weight:bold;")
        title_row.addWidget(cp_title)
        title_row.addStretch()
        control_layout.addLayout(title_row)

        bp_label = QtWidgets.QLabel("Batteries Per Package")
        bp_label.setStyleSheet("font-weight:bold;")
        control_layout.addWidget(bp_label)

        self.combo_expected = QtWidgets.QComboBox()
        for n in BATTERY_OPTIONS:
            text = f"{n} Battery" if n == 1 else f"{n} Batteries"
            self.combo_expected.addItem(text, n)
        self.combo_expected.setCurrentIndex(2)  # 4 batteries
        self.combo_expected.setStyleSheet("""
            QComboBox {
                padding:8px 10px;
                border-radius:8px;
                border:1px solid #cbd5e1;
                font-size:14px;
            }
        """)
        control_layout.addWidget(self.combo_expected)

        hint_lbl = QtWidgets.QLabel("Select the expected number of batteries in each package")
        hint_lbl.setStyleSheet("color:#6b7280;")
        control_layout.addWidget(hint_lbl)

        self.btn_start = QtWidgets.QPushButton("▶  Start Inspection")
        self.btn_start.setMinimumHeight(48)
        self.btn_start.setStyleSheet("""
            QPushButton {
                background:#16a34a; color:white; font-size:16px;
                font-weight:bold; border:none; border-radius:10px;
            }
            QPushButton:disabled {
                background:#9ca3af;
            }
        """)
        control_layout.addWidget(self.btn_start)

        self.btn_stop = QtWidgets.QPushButton("■  Stop")
        self.btn_stop.setMinimumHeight(40)
        self.btn_stop.setEnabled(False)
        self.btn_stop.setStyleSheet("""
            QPushButton {
                background:#ef4444; color:white; font-size:15px;
                font-weight:bold; border:none; border-radius:10px;
            }
            QPushButton:disabled {
                background:#fecaca; color:#7f1d1d;
            }
        """)
        control_layout.addWidget(self.btn_stop)

        self.btn_stop_alarm = QtWidgets.QPushButton("⏹  Stop Alarm")
        self.btn_stop_alarm.setMinimumHeight(40)
        self.btn_stop_alarm.setEnabled(False)
        self.btn_stop_alarm.setStyleSheet("""
            QPushButton {
                background:#b91c1c; color:white; font-size:15px;
                font-weight:bold; border:none; border-radius:10px;
            }
            QPushButton:disabled {
                background:#fee2e2; color:#991b1b;
            }
        """)
        control_layout.addWidget(self.btn_stop_alarm)

        self.btn_reset = QtWidgets.QPushButton("↻  Reset Counters")
        self.btn_reset.setMinimumHeight(36)
        self.btn_reset.setStyleSheet("""
            QPushButton {
                background:white; color:#111827; font-size:14px;
                border:1px solid #e5e7eb; border-radius:10px;
            }
        """)
        control_layout.addWidget(self.btn_reset)

        warning_frame = QtWidgets.QFrame()
        warning_frame.setStyleSheet("background:#fee2e2; border-radius:10px;")
        warning_layout = QtWidgets.QVBoxLayout(warning_frame)
        warning_layout.setContentsMargins(10, 8, 10, 8)
        self.warning_title = QtWidgets.QLabel("⚠ Battery Count Mismatch!")
        self.warning_title.setStyleSheet("font-weight:bold; color:#b91c1c;")
        self.warning_msg = QtWidgets.QLabel("Please check the package and resolve the issue.")
        self.warning_msg.setStyleSheet("color:#b91c1c;")
        warning_layout.addWidget(self.warning_title)
        warning_layout.addWidget(self.warning_msg)
        warning_frame.setVisible(False)
        control_layout.addWidget(warning_frame)
        self.warning_frame = warning_frame

        how_frame = QtWidgets.QFrame()
        how_frame.setStyleSheet("background:#eff6ff; border-radius:10px;")
        how_layout = QtWidgets.QVBoxLayout(how_frame)
        how_layout.setContentsMargins(10, 8, 10, 8)
        how_title = QtWidgets.QLabel("How it works:")
        how_title.setStyleSheet("font-weight:bold;")
        how_layout.addWidget(how_title)
        steps = QtWidgets.QLabel(
            "• Select batteries per package\n"
            "• Click Start to begin inspection\n"
            "• System detects and verifies each package\n"
            "• If mismatch detected, alarm sounds"
        )
        steps.setStyleSheet("color:#1f2937;")
        steps.setAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignTop)
        how_layout.addWidget(steps)
        control_layout.addWidget(how_frame)

        control_layout.addStretch()
        main_layout.addWidget(control_card, 1)

        # Right side: header + stats + results
        right_layout = QtWidgets.QVBoxLayout()
        right_layout.setSpacing(12)
        main_layout.addLayout(right_layout, 2)

        header_box = QtWidgets.QVBoxLayout()
        title_lbl = QtWidgets.QLabel("Battery Package Inspection")
        title_lbl.setStyleSheet("font-size:20px; font-weight:bold;")
        title_lbl.setAlignment(QtCore.Qt.AlignCenter)
        header_box.addWidget(title_lbl)

        subtitle = QtWidgets.QLabel("Automated Quantity Verification System")
        subtitle.setStyleSheet("color:#6b7280;")
        subtitle.setAlignment(QtCore.Qt.AlignCenter)
        header_box.addWidget(subtitle)
        right_layout.addLayout(header_box)

        stats_card = QtWidgets.QFrame()
        stats_card.setStyleSheet("background:white; border-radius:16px;")
        stats_layout = QtWidgets.QHBoxLayout(stats_card)
        stats_layout.setContentsMargins(16, 12, 16, 12)
        stats_layout.setSpacing(12)

        def make_stat(color_bg, title, unit):
            frame = QtWidgets.QFrame()
            frame.setStyleSheet(f"background:{color_bg}; border-radius:12px;")
            v = QtWidgets.QVBoxLayout(frame)
            v.setContentsMargins(12, 10, 12, 10)
            t = QtWidgets.QLabel(title)
            t.setStyleSheet("color:#6b7280;")
            v.addWidget(t)
            val = QtWidgets.QLabel("0")
            val.setStyleSheet("font-size:20px; font-weight:bold;")
            v.addWidget(val)
            u = QtWidgets.QLabel(unit)
            u.setStyleSheet("color:#9ca3af;")
            v.addWidget(u)
            v.addStretch()
            return frame, val

        self.stat_expected_frame, self.stat_expected_val = make_stat("#f9fafb", "Expected", "batteries/pkg")
        self.stat_total_frame, self.stat_total_val = make_stat("#eff6ff", "Total", "inspected")
        self.stat_passed_frame, self.stat_passed_val = make_stat("#ecfdf3", "Passed", "")
        self.stat_failed_frame, self.stat_failed_val = make_stat("#fef2f2", "Failed", "defects")

        stats_layout.addWidget(self.stat_expected_frame)
        stats_layout.addWidget(self.stat_total_frame)
        stats_layout.addWidget(self.stat_passed_frame)
        stats_layout.addWidget(self.stat_failed_frame)
        right_layout.addWidget(stats_card)

        # results area
        results_card = QtWidgets.QFrame()
        results_card.setStyleSheet("background:white; border-radius:16px;")
        results_layout = QtWidgets.QVBoxLayout(results_card)
        results_layout.setContentsMargins(16, 12, 16, 12)
        results_layout.setSpacing(8)

        res_title = QtWidgets.QLabel("Inspection Results")
        res_title.setStyleSheet("font-weight:bold;")
        results_layout.addWidget(res_title)

        self.empty_label = QtWidgets.QLabel('No inspections yet\nClick "Start Inspection" to begin')
        self.empty_label.setAlignment(QtCore.Qt.AlignCenter)
        self.empty_label.setStyleSheet("color:#9ca3af;")
        results_layout.addWidget(self.empty_label, 1)

        self.scroll = QtWidgets.QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll_content = QtWidgets.QWidget()
        self.scroll.setWidget(self.scroll_content)
        self.cards_layout = QtWidgets.QGridLayout(self.scroll_content)
        self.cards_layout.setContentsMargins(4, 4, 4, 4)
        self.cards_layout.setSpacing(12)
        results_layout.addWidget(self.scroll, 1)
        right_layout.addWidget(results_card, 3)

        # connections
        self.btn_start.clicked.connect(self.start_inspection)
        self.btn_stop.clicked.connect(self.stop_inspection)
        self.btn_stop_alarm.clicked.connect(self.stop_alarm)
        self.btn_reset.clicked.connect(self.reset_counters)

        self.update_expected_stat()

    # --- control logic ---

    def update_expected_stat(self):
        expected = self.combo_expected.currentData()
        self.stat_expected_val.setText(str(expected))

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
            QtWidgets.QMessageBox.critical(self, "Error", "auto_capture.py not found at expected path.")
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
        self.proc.terminate()
        try:
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
        # clear cards
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
