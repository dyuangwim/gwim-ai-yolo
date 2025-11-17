#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
示例：PyQt5 GUI + 检测线程 + 持续蜂鸣器控制

将此文件作为参考或直接替换你的 GUI 文件。根据你的硬件替换 BuzzerController 中的具体蜂鸣实现。
"""

import sys
import time
import threading
import platform

from PyQt5.QtCore import pyqtSignal, QThread, QObject, Qt
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QLabel, QVBoxLayout, QWidget


class BuzzerController:
    """
    负责管理蜂鸣器：start_buzz() 启动持续蜂鸣直到 stop_buzz()。
    这里提供了跨平台回退：在没有实际硬件时打印日志或用 winsound (Windows) 简单发声。
    在真实硬件上，请在这里集成你的 GPIO 或串口蜂鸣实现。
    """
    def __init__(self):
        self._thread = None
        self._stop_event = threading.Event()
        self._lock = threading.Lock()

    def _buzz_loop(self):
        # 持续蜂鸣循环：在这里放入实际驱动代码
        # 这里以打印与 platform-dependent beep 为示例（非真实硬件）
        if platform.system() == "Windows":
            try:
                import winsound
            except Exception:
                winsound = None
        else:
            winsound = None

        while not self._stop_event.is_set():
            # TODO: 将下面的打印/短促 beep 替换为硬件控制代码（如 GPIO 输出或串口指令）
            if winsound:
                winsound.Beep(2000, 200)  # 200ms beep
            else:
                # 在 Linux 上可尝试调用外部工具（如果可用），此处以打印为示例
                print("[BUZZER] buzz")
                # 模拟持续短促间隔
                time.sleep(0.2)
            # 给出短间隔，避免占用全部 CPU
            time.sleep(0.1)

    def start_buzz(self):
        with self._lock:
            # 如果已经在蜂鸣，什么都不做
            if self._thread and self._thread.is_alive():
                return
            self._stop_event.clear()
            self._thread = threading.Thread(target=self._buzz_loop, daemon=True)
            self._thread.start()
            print("[BUZZER] started")

    def stop_buzz(self):
        with self._lock:
            if not (self._thread and self._thread.is_alive()):
                self._stop_event.clear()
                return
            self._stop_event.set()
            self._thread.join(timeout=1.0)
            if self._thread.is_alive():
                print("[BUZZER] thread did not stop within timeout")
            else:
                print("[BUZZER] stopped")
            # reset thread handle
            self._thread = None
            self._stop_event.clear()


class DetectorWorker(QThread):
    """
    模拟检测工作线程。真实项目中把检测逻辑放在 run() 中并在触发 NG 时 emit ng_detected。
    stop() 方法用于安全地结束线程。
    """
    ng_detected = pyqtSignal()
    ok_detected = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._running = threading.Event()
        self._running.clear()
        self._simulate_counter = 0

    def run(self):
        self._running.set()
        print("[DETECTOR] started")
        try:
            while self._running.is_set():
                # TODO: 在这里放应有的检测逻辑，替换下面的模拟代码
                time.sleep(0.2)
                self._simulate_counter += 1
                # 每 20 次模拟一次 NG
                if self._simulate_counter % 20 == 0:
                    print("[DETECTOR] simulated NG")
                    self.ng_detected.emit()
                else:
                    # 可选：emit ok_detected
                    pass
        finally:
            print("[DETECTOR] stopped")

    def stop(self):
        self._running.clear()
        # 等待线程退出由外部 join，由 Qt 框架管理


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Factory Test - 示例")
        self.setGeometry(200, 200, 400, 200)

        # UI
        self.start_btn = QPushButton("Start")
        self.stop_btn = QPushButton("Stop")
        self.status_label = QLabel("Idle")
        self.status_label.setAlignment(Qt.AlignCenter)

        layout = QVBoxLayout()
        layout.addWidget(self.status_label)
        layout.addWidget(self.start_btn)
        layout.addWidget(self.stop_btn)

        central = QWidget()
        central.setLayout(layout)
        self.setCentralWidget(central)

        # 控件行为
        self.start_btn.clicked.connect(self.on_start)
        self.stop_btn.clicked.connect(self.on_stop)

        # 逻辑对象
        self.detector = None  # will be DetectorWorker instance
        self.buzzer = BuzzerController()

        self._is_running = False

        # 初始按钮状态
        self.update_buttons()

    def update_buttons(self):
        self.start_btn.setEnabled(not self._is_running)
        self.stop_btn.setEnabled(self._is_running)

    def on_start(self):
        # Start: 保证每次都用一个干净的 detector，并且重置 buzzer 状态
        if self._is_running:
            return
        # Ensure buzzer is stopped/reset before start
        self.buzzer.stop_buzz()

        # Create a new detector thread each start,避免重用已终止的线程对象
        self.detector = DetectorWorker()
        self.detector.ng_detected.connect(self.handle_ng)
        # 如果你需要处理 OK 也可以连接：
        # self.detector.ok_detected.connect(self.handle_ok)

        self.detector.start()

        self._is_running = True
        self.status_label.setText("Running")
        self.update_buttons()
        print("[MAIN] started test")

    def on_stop(self):
        # Stop：停止检测并停止蜂鸣
        if not self._is_running:
            return

        # Stop detector
        if self.detector:
            try:
                self.detector.stop()
                # 等待线程优雅退出
                self.detector.wait(1000)  # 1000 ms timeout
            except Exception as e:
                print("[MAIN] detector stop error:", e)
            self.detector = None

        # Stop buzzer
        self.buzzer.stop_buzz()

        self._is_running = False
        self.status_label.setText("Stopped")
        self.update_buttons()
        print("[MAIN] stopped test")

    def handle_ng(self):
        # NG 触发时启动持续蜂鸣（直到用户按 Stop）
        print("[MAIN] NG detected -> start buzzer")
        self.status_label.setText("NG")
        # 启动持续蜂鸣（内部会忽略重复 start 调用）
        self.buzzer.start_buzz()

    def closeEvent(self, event):
        # 窗口关闭时确保所有子线程、硬件清理
        self.on_stop()
        super().closeEvent(event)


def main():
    app = QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
