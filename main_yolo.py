"""
人物跟踪视频处理器 - GUI主程序
使用YOLO检测 + 轻量跟踪器的混合策略，降低CPU使用率
"""

import sys
import os
from pathlib import Path

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QFileDialog, QProgressBar, QSpinBox,
    QGroupBox, QComboBox, QCheckBox, QMessageBox, QFrame
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap

import cv2
import numpy as np

from yolo_detector import YOLODetector
from tracker import TrackerType
from video_processor import VideoProcessor


class ProcessingThread(QThread):
    """视频处理线程"""
    progress = pyqtSignal(int, int, dict)  # current, total, stats
    frame_ready = pyqtSignal(np.ndarray)   # 预览帧
    finished = pyqtSignal(dict)            # 完成信号
    error = pyqtSignal(str)                # 错误信号

    def __init__(self, processor: VideoProcessor, input_path: str,
                 output_path: str, skip_frames: int):
        super().__init__()
        self.processor = processor
        self.input_path = input_path
        self.output_path = output_path
        self.skip_frames = skip_frames

    def run(self):
        try:
            stats = self.processor.process_video(
                input_path=self.input_path,
                output_path=self.output_path,
                progress_callback=self._on_progress,
                preview_callback=self._on_frame,
                skip_frames=self.skip_frames
            )
            self.finished.emit(stats)
        except Exception as e:
            self.error.emit(str(e))

    def _on_progress(self, current, total, stats):
        self.progress.emit(current, total, stats)

    def _on_frame(self, frame):
        self.frame_ready.emit(frame)

    def stop(self):
        self.processor.stop()


class MainWindow(QMainWindow):
    """主窗口"""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("人物跟踪视频处理器 (YOLO + 轻量跟踪)")
        self.setMinimumSize(900, 700)

        # 初始化变量
        self.input_path = None
        self.output_path = None
        self.detector = None
        self.processor = None
        self.processing_thread = None

        self._setup_ui()

    def _setup_ui(self):
        """设置界面"""
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)

        # 文件选择区域
        file_group = QGroupBox("视频文件")
        file_layout = QVBoxLayout(file_group)

        # 输入文件
        input_layout = QHBoxLayout()
        self.input_label = QLabel("输入视频: 未选择")
        self.input_label.setStyleSheet("color: gray;")
        input_btn = QPushButton("选择输入视频")
        input_btn.clicked.connect(self._select_input)
        input_layout.addWidget(self.input_label, 1)
        input_layout.addWidget(input_btn)
        file_layout.addLayout(input_layout)

        # 输出文件
        output_layout = QHBoxLayout()
        self.output_label = QLabel("输出视频: 未选择")
        self.output_label.setStyleSheet("color: gray;")
        output_btn = QPushButton("选择输出位置")
        output_btn.clicked.connect(self._select_output)
        output_layout.addWidget(self.output_label, 1)
        output_layout.addWidget(output_btn)
        file_layout.addLayout(output_layout)

        layout.addWidget(file_group)

        # 参数设置区域
        params_group = QGroupBox("处理参数")
        params_layout = QHBoxLayout(params_group)

        # 跟踪器类型
        tracker_layout = QVBoxLayout()
        tracker_layout.addWidget(QLabel("跟踪器类型:"))
        self.tracker_combo = QComboBox()
        self.tracker_combo.addItems(["CSRT (推荐)", "KCF (快速)", "MOSSE (最快)"])
        tracker_layout.addWidget(self.tracker_combo)
        params_layout.addLayout(tracker_layout)

        # 重新检测间隔
        redetect_layout = QVBoxLayout()
        redetect_layout.addWidget(QLabel("YOLO重检间隔(帧):"))
        self.redetect_spin = QSpinBox()
        self.redetect_spin.setRange(10, 120)
        self.redetect_spin.setValue(30)
        self.redetect_spin.setToolTip("每隔多少帧使用YOLO重新检测")
        redetect_layout.addWidget(self.redetect_spin)
        params_layout.addLayout(redetect_layout)

        # 跳帧设置
        skip_layout = QVBoxLayout()
        skip_layout.addWidget(QLabel("跳帧数(省CPU):"))
        self.skip_spin = QSpinBox()
        self.skip_spin.setRange(0, 5)
        self.skip_spin.setValue(0)
        self.skip_spin.setToolTip("跳过处理的帧数，0表示处理每一帧")
        skip_layout.addWidget(self.skip_spin)
        params_layout.addLayout(skip_layout)

        # 预览开关
        preview_layout = QVBoxLayout()
        preview_layout.addWidget(QLabel("实时预览:"))
        self.preview_check = QCheckBox("启用")
        self.preview_check.setChecked(True)
        self.preview_check.setToolTip("关闭可进一步降低CPU使用")
        preview_layout.addWidget(self.preview_check)
        params_layout.addLayout(preview_layout)

        layout.addWidget(params_group)

        # 预览区域
        preview_group = QGroupBox("视频预览")
        preview_layout = QVBoxLayout(preview_group)

        self.preview_label = QLabel("选择视频文件后开始处理")
        self.preview_label.setAlignment(Qt.AlignCenter)
        self.preview_label.setMinimumHeight(360)
        self.preview_label.setStyleSheet(
            "background-color: #1a1a1a; color: #666; border: 1px solid #333;"
        )
        preview_layout.addWidget(self.preview_label)

        layout.addWidget(preview_group, 1)

        # 进度和统计
        progress_group = QGroupBox("处理进度")
        progress_layout = QVBoxLayout(progress_group)

        self.progress_bar = QProgressBar()
        self.progress_bar.setTextVisible(True)
        progress_layout.addWidget(self.progress_bar)

        self.stats_label = QLabel("就绪")
        self.stats_label.setAlignment(Qt.AlignCenter)
        progress_layout.addWidget(self.stats_label)

        layout.addWidget(progress_group)

        # 控制按钮
        btn_layout = QHBoxLayout()

        self.start_btn = QPushButton("开始处理")
        self.start_btn.setEnabled(False)
        self.start_btn.clicked.connect(self._start_processing)
        self.start_btn.setStyleSheet(
            "QPushButton { background-color: #2e7d32; color: white; padding: 10px; }"
            "QPushButton:disabled { background-color: #555; }"
        )
        btn_layout.addWidget(self.start_btn)

        self.stop_btn = QPushButton("停止")
        self.stop_btn.setEnabled(False)
        self.stop_btn.clicked.connect(self._stop_processing)
        self.stop_btn.setStyleSheet(
            "QPushButton { background-color: #c62828; color: white; padding: 10px; }"
            "QPushButton:disabled { background-color: #555; }"
        )
        btn_layout.addWidget(self.stop_btn)

        layout.addLayout(btn_layout)

    def _select_input(self):
        """选择输入视频"""
        path, _ = QFileDialog.getOpenFileName(
            self, "选择视频文件", "",
            "视频文件 (*.mp4 *.avi *.mkv *.mov *.wmv);;所有文件 (*)"
        )
        if path:
            self.input_path = path
            self.input_label.setText(f"输入视频: {Path(path).name}")
            self.input_label.setStyleSheet("color: white;")

            # 自动生成输出路径
            if not self.output_path:
                stem = Path(path).stem
                suffix = Path(path).suffix
                self.output_path = str(Path(path).parent / f"{stem}_tracked{suffix}")
                self.output_label.setText(f"输出视频: {Path(self.output_path).name}")
                self.output_label.setStyleSheet("color: #aaa;")

            self._update_start_button()

    def _select_output(self):
        """选择输出位置"""
        path, _ = QFileDialog.getSaveFileName(
            self, "保存输出视频", "",
            "MP4视频 (*.mp4);;AVI视频 (*.avi);;所有文件 (*)"
        )
        if path:
            self.output_path = path
            self.output_label.setText(f"输出视频: {Path(path).name}")
            self.output_label.setStyleSheet("color: white;")
            self._update_start_button()

    def _update_start_button(self):
        """更新开始按钮状态"""
        self.start_btn.setEnabled(
            self.input_path is not None and
            self.output_path is not None
        )

    def _get_tracker_type(self) -> TrackerType:
        """获取选择的跟踪器类型"""
        idx = self.tracker_combo.currentIndex()
        return [TrackerType.CSRT, TrackerType.KCF, TrackerType.MOSSE][idx]

    def _start_processing(self):
        """开始处理"""
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.progress_bar.setValue(0)
        self.stats_label.setText("初始化中...")

        # 初始化检测器 (第一次使用时)
        if self.detector is None:
            self.stats_label.setText("正在加载YOLO模型 (首次需下载)...")
            QApplication.processEvents()
            self.detector = YOLODetector()

        # 创建处理器
        self.processor = VideoProcessor(
            detector=self.detector,
            tracker_type=self._get_tracker_type(),
            redetect_interval=self.redetect_spin.value()
        )

        # 创建处理线程
        self.processing_thread = ProcessingThread(
            processor=self.processor,
            input_path=self.input_path,
            output_path=self.output_path,
            skip_frames=self.skip_spin.value()
        )

        self.processing_thread.progress.connect(self._on_progress)
        if self.preview_check.isChecked():
            self.processing_thread.frame_ready.connect(self._on_frame)
        self.processing_thread.finished.connect(self._on_finished)
        self.processing_thread.error.connect(self._on_error)

        self.processing_thread.start()

    def _stop_processing(self):
        """停止处理"""
        if self.processing_thread:
            self.processing_thread.stop()
            self.stats_label.setText("正在停止...")

    def _on_progress(self, current: int, total: int, stats: dict):
        """进度更新"""
        percent = int(current / total * 100) if total > 0 else 0
        self.progress_bar.setValue(percent)

        self.stats_label.setText(
            f"帧 {current}/{total} | "
            f"FPS: {stats['avg_fps']:.1f} | "
            f"YOLO帧: {stats['yolo_frames']} | "
            f"跟踪帧: {stats['tracker_frames']} | "
            f"累计人数: {stats.get('total_persons', 0)}"
        )

    def _on_frame(self, frame: np.ndarray):
        """预览帧更新"""
        # 缩放预览
        h, w = frame.shape[:2]
        max_h = self.preview_label.height() - 10
        scale = max_h / h
        new_w, new_h = int(w * scale), int(h * scale)
        resized = cv2.resize(frame, (new_w, new_h))

        # 转换为QImage
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qimg = QImage(rgb.data, w, h, w * ch, QImage.Format_RGB888)
        self.preview_label.setPixmap(QPixmap.fromImage(qimg))

    def _on_finished(self, stats: dict):
        """处理完成"""
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.progress_bar.setValue(100)

        QMessageBox.information(
            self, "处理完成",
            f"视频处理完成!\n\n"
            f"总帧数: {stats['total_frames']}\n"
            f"YOLO检测帧: {stats['yolo_frames']}\n"
            f"跟踪器帧: {stats['tracker_frames']}\n"
            f"累计跟踪人数: {stats.get('total_persons', 0)}\n"
            f"平均FPS: {stats['avg_fps']:.2f}\n\n"
            f"输出已保存到:\n{self.output_path}"
        )

    def _on_error(self, error: str):
        """处理错误"""
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        QMessageBox.critical(self, "错误", f"处理出错:\n{error}")

    def closeEvent(self, event):
        """窗口关闭时停止处理"""
        if self.processing_thread and self.processing_thread.isRunning():
            self.processing_thread.stop()
            self.processing_thread.wait()
        event.accept()


def main():
    app = QApplication(sys.argv)
    app.setStyle('Fusion')

    # 设置暗色主题
    from PyQt5.QtGui import QPalette, QColor
    palette = QPalette()
    palette.setColor(QPalette.Window, QColor(53, 53, 53))
    palette.setColor(QPalette.WindowText, Qt.white)
    palette.setColor(QPalette.Base, QColor(25, 25, 25))
    palette.setColor(QPalette.AlternateBase, QColor(53, 53, 53))
    palette.setColor(QPalette.ToolTipBase, Qt.white)
    palette.setColor(QPalette.ToolTipText, Qt.white)
    palette.setColor(QPalette.Text, Qt.white)
    palette.setColor(QPalette.Button, QColor(53, 53, 53))
    palette.setColor(QPalette.ButtonText, Qt.white)
    palette.setColor(QPalette.BrightText, Qt.red)
    palette.setColor(QPalette.Highlight, QColor(42, 130, 218))
    palette.setColor(QPalette.HighlightedText, Qt.black)
    app.setPalette(palette)

    window = MainWindow()
    window.show()

    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
