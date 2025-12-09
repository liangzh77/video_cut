"""
视频剪切工具 - 时间范围选择 + 区域裁剪
"""

import sys
import time
import cv2
import numpy as np
from pathlib import Path
from datetime import datetime

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QFileDialog, QSlider, QGroupBox, QMessageBox,
    QProgressDialog, QSizePolicy, QComboBox, QButtonGroup, QRadioButton,
    QListWidget, QListWidgetItem, QSplitter, QFrame
)
from PyQt5.QtCore import Qt, QRect, QPoint, pyqtSignal, QTimer
from PyQt5.QtGui import QImage, QPixmap, QPainter, QColor, QFont, QPen, QCursor


class RangeSlider(QWidget):
    """三滑块：起点、终点、预览"""

    startChanged = pyqtSignal(int)
    endChanged = pyqtSignal(int)
    previewChanged = pyqtSignal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(60)

        self._min = 0
        self._max = 100
        self._start = 0
        self._end = 100
        self._preview = 0

        self._pressing_start = False
        self._pressing_end = False
        self._pressing_preview = False

        self.setMouseTracking(True)

    def setRange(self, min_val, max_val):
        self._min = min_val
        self._max = max_val
        self._start = min_val
        self._end = max_val
        self._preview = min_val
        self.update()

    def setStart(self, val):
        self._start = max(self._min, min(val, self._end - 1))
        self.update()
        self.startChanged.emit(self._start)

    def setEnd(self, val):
        self._end = min(self._max, max(val, self._start + 1))
        self.update()
        self.endChanged.emit(self._end)

    def setPreview(self, val):
        self._preview = max(self._min, min(val, self._max))
        self.update()
        self.previewChanged.emit(self._preview)

    def start(self):
        return self._start

    def end(self):
        return self._end

    def preview(self):
        return self._preview

    def _pos_to_value(self, x):
        margin = 15
        width = self.width() - 2 * margin
        if width <= 0:
            return self._min
        ratio = (x - margin) / width
        return int(self._min + ratio * (self._max - self._min))

    def _value_to_pos(self, val):
        margin = 15
        width = self.width() - 2 * margin
        if self._max == self._min:
            return margin
        ratio = (val - self._min) / (self._max - self._min)
        return int(margin + ratio * width)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        margin = 15
        track_height = 8
        handle_radius = 10

        # 轨道背景
        track_y = 25
        track_rect = QRect(margin, track_y, self.width() - 2 * margin, track_height)
        painter.setBrush(QColor(60, 60, 60))
        painter.setPen(Qt.NoPen)
        painter.drawRoundedRect(track_rect, 4, 4)

        # 选中区域
        start_x = self._value_to_pos(self._start)
        end_x = self._value_to_pos(self._end)
        selected_rect = QRect(start_x, track_y, end_x - start_x, track_height)
        painter.setBrush(QColor(66, 133, 244, 150))
        painter.drawRoundedRect(selected_rect, 4, 4)

        # 预览位置线
        preview_x = self._value_to_pos(self._preview)
        painter.setPen(QPen(QColor(255, 255, 0), 2))
        painter.drawLine(preview_x, track_y - 5, preview_x, track_y + track_height + 5)

        # 预览滑块 (三角形)
        painter.setBrush(QColor(255, 255, 0))
        painter.setPen(Qt.NoPen)
        triangle = [
            QPoint(preview_x, track_y - 8),
            QPoint(preview_x - 8, track_y - 18),
            QPoint(preview_x + 8, track_y - 18)
        ]
        painter.drawPolygon(*triangle)

        # 起点滑块
        painter.setBrush(QColor(76, 175, 80))
        painter.drawEllipse(start_x - handle_radius, track_y + track_height + 5,
                           handle_radius * 2, handle_radius * 2)

        # 终点滑块
        painter.setBrush(QColor(244, 67, 54))
        painter.drawEllipse(end_x - handle_radius, track_y + track_height + 5,
                           handle_radius * 2, handle_radius * 2)

        # 标签
        painter.setPen(QColor(200, 200, 200))
        painter.setFont(QFont("Arial", 8))
        painter.drawText(start_x - 12, self.height() - 2, "起点")
        painter.drawText(end_x - 12, self.height() - 2, "终点")
        painter.drawText(preview_x - 12, 8, "预览")

    def mousePressEvent(self, event):
        x = event.x()
        y = event.y()
        start_x = self._value_to_pos(self._start)
        end_x = self._value_to_pos(self._end)
        preview_x = self._value_to_pos(self._preview)

        track_y = 25

        # 判断点击哪个滑块
        if abs(x - preview_x) < 15 and y < track_y:
            self._pressing_preview = True
        elif abs(x - start_x) < 15 and y > track_y:
            self._pressing_start = True
        elif abs(x - end_x) < 15 and y > track_y:
            self._pressing_end = True
        else:
            # 点击轨道区域，移动预览
            self._pressing_preview = True
            val = self._pos_to_value(x)
            self.setPreview(val)

    def mouseReleaseEvent(self, event):
        self._pressing_start = False
        self._pressing_end = False
        self._pressing_preview = False

    def mouseMoveEvent(self, event):
        if self._pressing_start:
            val = self._pos_to_value(event.x())
            self.setStart(val)
        elif self._pressing_end:
            val = self._pos_to_value(event.x())
            self.setEnd(val)
        elif self._pressing_preview:
            val = self._pos_to_value(event.x())
            self.setPreview(val)


class CropLabel(QLabel):
    """支持区域选择的预览标签"""

    cropChanged = pyqtSignal(tuple)  # (x, y, w, h) 归一化坐标

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMouseTracking(True)

        # 裁剪区域 (归一化 0-1)
        self._crop_x = 0.0
        self._crop_y = 0.0
        self._crop_w = 1.0
        self._crop_h = 1.0

        # 宽高比约束 (None = 自由, 否则为 w/h 比值)
        self._aspect_ratio = None
        self._video_aspect = 1.0  # 视频本身的宽高比

        # 拖动状态
        self._dragging = None  # 'tl', 'tr', 'bl', 'br', 't', 'b', 'l', 'r', 'move'
        self._drag_start = None
        self._drag_crop_start = None

        # 图像尺寸
        self._img_rect = QRect()

    def setAspectRatio(self, ratio):
        """设置宽高比约束，None为自由模式"""
        self._aspect_ratio = ratio
        if ratio is not None:
            # 应用宽高比到当前裁剪区域
            self._apply_aspect_ratio()

    def setVideoSize(self, width, height):
        """设置视频尺寸，用于计算实际宽高比"""
        if height > 0:
            self._video_aspect = width / height

    def _apply_aspect_ratio(self):
        """应用宽高比约束到当前裁剪区域"""
        if self._aspect_ratio is None:
            return

        # 目标宽高比（考虑视频本身的宽高比）
        target_ratio = self._aspect_ratio / self._video_aspect

        current_ratio = self._crop_w / self._crop_h if self._crop_h > 0 else 1

        if current_ratio > target_ratio:
            # 当前太宽，缩小宽度
            new_w = self._crop_h * target_ratio
            self._crop_x += (self._crop_w - new_w) / 2
            self._crop_w = new_w
        else:
            # 当前太高，缩小高度
            new_h = self._crop_w / target_ratio
            self._crop_y += (self._crop_h - new_h) / 2
            self._crop_h = new_h

        # 确保在边界内
        self._crop_x = max(0, min(self._crop_x, 1 - self._crop_w))
        self._crop_y = max(0, min(self._crop_y, 1 - self._crop_h))

        self.update()
        self.cropChanged.emit(self.getCrop())

    def setCrop(self, x, y, w, h, anchor=None):
        """
        设置裁剪区域
        anchor: 固定的锚点 'tl', 'tr', 'bl', 'br', 't', 'b', 'l', 'r', None
                表示该点位置固定不动
        """
        min_size = 0.05

        if self._aspect_ratio is None:
            # 自由模式
            if anchor is None:
                # 移动模式：保持尺寸不变，只调整位置
                w = max(min_size, min(w, 1))
                h = max(min_size, min(h, 1))
                # 限制位置，确保不超出边界且不改变尺寸
                self._crop_x = max(0, min(x, 1 - w))
                self._crop_y = max(0, min(y, 1 - h))
                self._crop_w = w
                self._crop_h = h
            else:
                # 调整大小模式
                self._crop_x = max(0, min(x, 1 - min_size))
                self._crop_y = max(0, min(y, 1 - min_size))
                self._crop_w = max(min_size, min(w, 1 - self._crop_x))
                self._crop_h = max(min_size, min(h, 1 - self._crop_y))
        else:
            # 固定宽高比模式
            target_ratio = self._aspect_ratio / self._video_aspect

            if anchor is None:
                # 移动模式：保持尺寸不变，只调整位置
                # 限制位置，确保不超出边界
                self._crop_x = max(0, min(x, 1 - self._crop_w))
                self._crop_y = max(0, min(y, 1 - self._crop_h))
                # 尺寸不变
            else:
                # 调整大小模式
                # 限制最小尺寸
                w = max(min_size, w)
                h = max(min_size, h)

                # 保持宽高比（以较小的变化为准）
                if w / h > target_ratio:
                    w = h * target_ratio
                else:
                    h = w / target_ratio

                # 根据锚点确定哪些边界是固定的
                if anchor in ('tl', 't', 'tr'):
                    # 上边固定，检查下边界
                    if y + h > 1:
                        h = 1 - y
                        w = h * target_ratio
                if anchor in ('bl', 'b', 'br'):
                    # 下边固定，检查上边界
                    if y < 0:
                        return
                if anchor in ('tl', 'l', 'bl'):
                    # 左边固定，检查右边界
                    if x + w > 1:
                        w = 1 - x
                        h = w / target_ratio
                if anchor in ('tr', 'r', 'br'):
                    # 右边固定，检查左边界
                    if x < 0:
                        return

                # 检查尺寸是否有效
                if w < min_size or h < min_size:
                    return

                # 最终边界检查
                if x < 0 or y < 0 or x + w > 1 or y + h > 1:
                    return

                self._crop_x = x
                self._crop_y = y
                self._crop_w = w
                self._crop_h = h

        self.update()
        self.cropChanged.emit(self.getCrop())

    def getCrop(self):
        return (self._crop_x, self._crop_y, self._crop_w, self._crop_h)

    def resetCrop(self):
        self.setCrop(0, 0, 1, 1)

    def setPixmap(self, pixmap):
        super().setPixmap(pixmap)
        self._updateImgRect()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._updateImgRect()

    def _updateImgRect(self):
        """更新图像在label中的实际位置"""
        pixmap = self.pixmap()
        if pixmap and not pixmap.isNull():
            pw, ph = pixmap.width(), pixmap.height()
            lw, lh = self.width(), self.height()
            # 居中
            x = (lw - pw) // 2
            y = (lh - ph) // 2
            self._img_rect = QRect(x, y, pw, ph)
        else:
            self._img_rect = QRect()

    def _crop_to_widget(self):
        """裁剪区域转widget坐标"""
        if self._img_rect.isEmpty():
            return QRect()
        x = self._img_rect.x() + self._crop_x * self._img_rect.width()
        y = self._img_rect.y() + self._crop_y * self._img_rect.height()
        w = self._crop_w * self._img_rect.width()
        h = self._crop_h * self._img_rect.height()
        return QRect(int(x), int(y), int(w), int(h))

    def _widget_to_crop(self, x, y):
        """widget坐标转归一化坐标"""
        if self._img_rect.isEmpty() or self._img_rect.width() == 0:
            return 0, 0
        nx = (x - self._img_rect.x()) / self._img_rect.width()
        ny = (y - self._img_rect.y()) / self._img_rect.height()
        return max(0, min(1, nx)), max(0, min(1, ny))

    def paintEvent(self, event):
        super().paintEvent(event)

        if self._img_rect.isEmpty():
            return

        painter = QPainter(self)
        crop_rect = self._crop_to_widget()

        # 半透明遮罩
        painter.setBrush(QColor(0, 0, 0, 120))
        painter.setPen(Qt.NoPen)

        # 上
        painter.drawRect(self._img_rect.x(), self._img_rect.y(),
                        self._img_rect.width(), crop_rect.y() - self._img_rect.y())
        # 下
        painter.drawRect(self._img_rect.x(), crop_rect.bottom(),
                        self._img_rect.width(), self._img_rect.bottom() - crop_rect.bottom())
        # 左
        painter.drawRect(self._img_rect.x(), crop_rect.y(),
                        crop_rect.x() - self._img_rect.x(), crop_rect.height())
        # 右
        painter.drawRect(crop_rect.right(), crop_rect.y(),
                        self._img_rect.right() - crop_rect.right(), crop_rect.height())

        # 裁剪边框
        painter.setPen(QPen(QColor(0, 255, 0), 2))
        painter.setBrush(Qt.NoBrush)
        painter.drawRect(crop_rect)

        # 四个角的拖动点
        handle_size = 10
        painter.setBrush(QColor(0, 255, 0))
        corners = [
            (crop_rect.topLeft(), 'tl'),
            (crop_rect.topRight(), 'tr'),
            (crop_rect.bottomLeft(), 'bl'),
            (crop_rect.bottomRight(), 'br'),
        ]
        for point, _ in corners:
            painter.drawRect(point.x() - handle_size // 2, point.y() - handle_size // 2,
                            handle_size, handle_size)

        # 四条边的中点
        painter.setBrush(QColor(0, 200, 0))
        edges = [
            (QPoint(crop_rect.center().x(), crop_rect.top()), 't'),
            (QPoint(crop_rect.center().x(), crop_rect.bottom()), 'b'),
            (QPoint(crop_rect.left(), crop_rect.center().y()), 'l'),
            (QPoint(crop_rect.right(), crop_rect.center().y()), 'r'),
        ]
        for point, _ in edges:
            painter.drawRect(point.x() - handle_size // 2, point.y() - handle_size // 2,
                            handle_size, handle_size)

    def mousePressEvent(self, event):
        if event.button() != Qt.LeftButton:
            return

        pos = event.pos()
        crop_rect = self._crop_to_widget()
        handle_size = 15

        # 检测点击位置
        corners = {
            'tl': crop_rect.topLeft(),
            'tr': crop_rect.topRight(),
            'bl': crop_rect.bottomLeft(),
            'br': crop_rect.bottomRight(),
        }
        edges = {
            't': QPoint(crop_rect.center().x(), crop_rect.top()),
            'b': QPoint(crop_rect.center().x(), crop_rect.bottom()),
            'l': QPoint(crop_rect.left(), crop_rect.center().y()),
            'r': QPoint(crop_rect.right(), crop_rect.center().y()),
        }

        for name, point in corners.items():
            if abs(pos.x() - point.x()) < handle_size and abs(pos.y() - point.y()) < handle_size:
                self._dragging = name
                break
        else:
            for name, point in edges.items():
                if abs(pos.x() - point.x()) < handle_size and abs(pos.y() - point.y()) < handle_size:
                    self._dragging = name
                    break
            else:
                if crop_rect.contains(pos):
                    self._dragging = 'move'

        if self._dragging:
            self._drag_start = pos
            self._drag_crop_start = (self._crop_x, self._crop_y, self._crop_w, self._crop_h)

    def mouseReleaseEvent(self, event):
        self._dragging = None
        self._drag_start = None

    def mouseMoveEvent(self, event):
        pos = event.pos()
        crop_rect = self._crop_to_widget()

        # 更新光标
        if not self._dragging:
            handle_size = 15
            corners = {'tl': crop_rect.topLeft(), 'tr': crop_rect.topRight(),
                      'bl': crop_rect.bottomLeft(), 'br': crop_rect.bottomRight()}
            edges = {'t': QPoint(crop_rect.center().x(), crop_rect.top()),
                    'b': QPoint(crop_rect.center().x(), crop_rect.bottom()),
                    'l': QPoint(crop_rect.left(), crop_rect.center().y()),
                    'r': QPoint(crop_rect.right(), crop_rect.center().y())}

            cursor = Qt.ArrowCursor
            for name, point in corners.items():
                if abs(pos.x() - point.x()) < handle_size and abs(pos.y() - point.y()) < handle_size:
                    if name in ('tl', 'br'):
                        cursor = Qt.SizeFDiagCursor
                    else:
                        cursor = Qt.SizeBDiagCursor
                    break
            else:
                for name, point in edges.items():
                    if abs(pos.x() - point.x()) < handle_size and abs(pos.y() - point.y()) < handle_size:
                        if name in ('t', 'b'):
                            cursor = Qt.SizeVerCursor
                        else:
                            cursor = Qt.SizeHorCursor
                        break
                else:
                    if crop_rect.contains(pos):
                        cursor = Qt.SizeAllCursor

            self.setCursor(cursor)

        # 拖动处理
        if self._dragging and self._drag_start and self._img_rect.width() > 0:
            dx = (pos.x() - self._drag_start.x()) / self._img_rect.width()
            dy = (pos.y() - self._drag_start.y()) / self._img_rect.height()

            ox, oy, ow, oh = self._drag_crop_start

            if self._dragging == 'move':
                self.setCrop(ox + dx, oy + dy, ow, oh)
            elif self._aspect_ratio is None:
                # 自由模式
                if self._dragging == 'tl':
                    self.setCrop(ox + dx, oy + dy, ow - dx, oh - dy)
                elif self._dragging == 'tr':
                    self.setCrop(ox, oy + dy, ow + dx, oh - dy)
                elif self._dragging == 'bl':
                    self.setCrop(ox + dx, oy, ow - dx, oh + dy)
                elif self._dragging == 'br':
                    self.setCrop(ox, oy, ow + dx, oh + dy)
                elif self._dragging == 't':
                    self.setCrop(ox, oy + dy, ow, oh - dy)
                elif self._dragging == 'b':
                    self.setCrop(ox, oy, ow, oh + dy)
                elif self._dragging == 'l':
                    self.setCrop(ox + dx, oy, ow - dx, oh)
                elif self._dragging == 'r':
                    self.setCrop(ox, oy, ow + dx, oh)
            else:
                # 固定宽高比模式
                target_ratio = self._aspect_ratio / self._video_aspect

                # 确定锚点（固定点是拖动点的对角/对边）
                anchor_map = {
                    'tl': 'br', 'tr': 'bl', 'bl': 'tr', 'br': 'tl',
                    't': 'b', 'b': 't', 'l': 'r', 'r': 'l'
                }
                anchor = anchor_map.get(self._dragging)

                if self._dragging in ('tl', 'tr', 'bl', 'br'):
                    # 角拖动：根据移动距离较大的方向决定尺寸
                    if abs(dx) > abs(dy):
                        if self._dragging in ('tl', 'bl'):
                            new_w = ow - dx
                        else:
                            new_w = ow + dx
                        new_h = new_w / target_ratio
                    else:
                        if self._dragging in ('tl', 'tr'):
                            new_h = oh - dy
                        else:
                            new_h = oh + dy
                        new_w = new_h * target_ratio

                    # 计算新位置（固定对角点）
                    if self._dragging == 'tl':
                        new_x = ox + ow - new_w
                        new_y = oy + oh - new_h
                    elif self._dragging == 'tr':
                        new_x = ox
                        new_y = oy + oh - new_h
                    elif self._dragging == 'bl':
                        new_x = ox + ow - new_w
                        new_y = oy
                    else:  # br
                        new_x = ox
                        new_y = oy

                    self.setCrop(new_x, new_y, new_w, new_h, anchor)

                elif self._dragging in ('t', 'b'):
                    if self._dragging == 't':
                        new_h = oh - dy
                        new_y = oy + oh - new_h
                    else:
                        new_h = oh + dy
                        new_y = oy
                    new_w = new_h * target_ratio
                    new_x = ox + (ow - new_w) / 2
                    self.setCrop(new_x, new_y, new_w, new_h, anchor)

                elif self._dragging in ('l', 'r'):
                    if self._dragging == 'l':
                        new_w = ow - dx
                        new_x = ox + ow - new_w
                    else:
                        new_w = ow + dx
                        new_x = ox
                    new_h = new_w / target_ratio
                    new_y = oy + (oh - new_h) / 2
                    self.setCrop(new_x, new_y, new_w, new_h, anchor)


class VideoCutWindow(QMainWindow):
    """视频剪切主窗口"""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("视频剪切工具")
        self.setMinimumSize(1000, 800)

        self.video_path = None
        self.cap = None
        self.total_frames = 0
        self.fps = 30
        self.width = 0
        self.height = 0

        # 播放相关
        self.play_timer = QTimer()
        self.play_timer.timeout.connect(self._on_play_timer)
        self.is_playing = False
        self.play_speed = 1.0
        self.play_range_only = False
        self.last_play_time = 0  # 上次播放的时间戳
        self.frame_skip = 1  # 跳帧数

        self._setup_ui()

    def _setup_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)

        # 左侧文件列表面板
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0, 0, 0, 0)

        # 文件夹选择
        folder_layout = QHBoxLayout()
        self.folder_label = QLabel("videos")
        self.folder_label.setStyleSheet("color: #aaa; font-size: 12px;")
        folder_btn = QPushButton("选择文件夹")
        folder_btn.clicked.connect(self._select_folder)
        folder_btn.setStyleSheet("QPushButton { color: white; padding: 8px 20px; font-size: 13px; background-color: #2196F3; } QPushButton:hover { background-color: #1976D2; }")
        folder_layout.addWidget(self.folder_label, 1)
        folder_layout.addWidget(folder_btn)
        left_layout.addLayout(folder_layout)

        # 源视频列表
        source_label = QLabel("源视频")
        source_label.setStyleSheet("color: #4CAF50; font-weight: bold; font-size: 16px; padding: 5px 0;")
        left_layout.addWidget(source_label)

        self.file_list = QListWidget()
        self.file_list.setStyleSheet("QListWidget { background-color: #2a2a2a; border: 1px solid #444; }"
                                      "QListWidget::item { padding: 8px; }"
                                      "QListWidget::item:selected { background-color: #4CAF50; }")
        self.file_list.itemClicked.connect(self._on_file_selected)
        left_layout.addWidget(self.file_list)

        # 生成视频列表
        generated_label = QLabel("生成视频（右键打开文件夹）")
        generated_label.setStyleSheet("color: #2196F3; font-weight: bold; font-size: 16px; padding: 5px 0;")
        left_layout.addWidget(generated_label)

        self.generated_list = QListWidget()
        self.generated_list.setStyleSheet("QListWidget { background-color: #2a2a2a; border: 1px solid #444; }"
                                           "QListWidget::item { padding: 8px; }"
                                           "QListWidget::item:selected { background-color: #4CAF50; }")
        self.generated_list.itemClicked.connect(self._on_generated_selected)
        self.generated_list.setContextMenuPolicy(Qt.CustomContextMenu)
        self.generated_list.customContextMenuRequested.connect(self._on_generated_right_click)
        left_layout.addWidget(self.generated_list)

        left_panel.setFixedWidth(250)
        main_layout.addWidget(left_panel)

        # 右侧主内容区
        right_panel = QWidget()
        layout = QVBoxLayout(right_panel)
        layout.setContentsMargins(5, 0, 0, 0)

        # 当前文件信息
        self.file_label = QLabel("未选择视频")
        self.file_label.setStyleSheet("color: gray;")
        layout.addWidget(self.file_label)

        # 初始化文件夹
        self.current_folder = Path(__file__).parent / "videos"
        if not self.current_folder.exists():
            self.current_folder.mkdir(exist_ok=True)
        self.generated_folder = self.current_folder / "生成"
        if not self.generated_folder.exists():
            self.generated_folder.mkdir(exist_ok=True)
        self._refresh_file_list()

        main_layout.addWidget(right_panel, 1)

        # 预览区域 (支持区域选择)
        preview_group = QGroupBox()
        preview_layout = QVBoxLayout(preview_group)

        self.preview_label = CropLabel()
        self.preview_label.setAlignment(Qt.AlignCenter)
        self.preview_label.setMinimumHeight(400)
        self.preview_label.setStyleSheet(
            "background-color: #1a1a1a; color: #666; border: 1px solid #333;"
        )
        self.preview_label.cropChanged.connect(self._on_crop_changed)
        preview_layout.addWidget(self.preview_label)

        # 裁剪信息和重置按钮
        crop_layout = QHBoxLayout()
        self.crop_info_label = QLabel("裁剪区域: 全部")
        reset_crop_btn = QPushButton("重置裁剪")
        reset_crop_btn.clicked.connect(self._reset_crop)
        crop_layout.addWidget(self.crop_info_label)
        crop_layout.addStretch()
        crop_layout.addWidget(reset_crop_btn)
        preview_layout.addLayout(crop_layout)

        layout.addWidget(preview_group, 1)

        # 时间信息
        time_layout = QHBoxLayout()
        self.start_time_label = QLabel("起点: 00:00:00")
        self.preview_time_label = QLabel("预览: 00:00:00")
        self.end_time_label = QLabel("终点: 00:00:00")
        self.duration_label = QLabel("时长: 00:00:00")
        self.start_time_label.setStyleSheet("color: #4CAF50; font-weight: bold;")
        self.preview_time_label.setStyleSheet("color: #FFEB3B; font-weight: bold;")
        self.end_time_label.setStyleSheet("color: #F44336; font-weight: bold;")
        self.duration_label.setStyleSheet("color: #2196F3; font-weight: bold;")
        time_layout.addWidget(self.start_time_label)
        time_layout.addWidget(self.preview_time_label)
        time_layout.addStretch()
        time_layout.addWidget(self.duration_label)
        time_layout.addStretch()
        time_layout.addWidget(self.end_time_label)
        layout.addLayout(time_layout)

        # 范围选择滑块
        slider_group = QGroupBox("时间线 (黄色=预览, 绿色=起点, 红色=终点)")
        slider_layout = QVBoxLayout(slider_group)

        self.range_slider = RangeSlider()
        self.range_slider.startChanged.connect(self._on_start_changed)
        self.range_slider.endChanged.connect(self._on_end_changed)
        self.range_slider.previewChanged.connect(self._on_preview_changed)
        slider_layout.addWidget(self.range_slider)

        layout.addWidget(slider_group)

        # 操作按钮 + 播放控制
        btn_layout = QHBoxLayout()

        blue_btn_style = "QPushButton { color: white; padding: 8px 20px; font-size: 13px; background-color: #2196F3; } QPushButton:disabled { background-color: #555; color: #888; } QPushButton:hover { background-color: #1976D2; }"

        self.set_start_btn = QPushButton("设为起点")
        self.set_start_btn.clicked.connect(self._set_preview_as_start)
        self.set_start_btn.setEnabled(False)
        self.set_start_btn.setStyleSheet(blue_btn_style)
        btn_layout.addWidget(self.set_start_btn)

        self.set_end_btn = QPushButton("设为终点")
        self.set_end_btn.clicked.connect(self._set_preview_as_end)
        self.set_end_btn.setEnabled(False)
        self.set_end_btn.setStyleSheet(blue_btn_style)
        btn_layout.addWidget(self.set_end_btn)

        self.export_btn = QPushButton("导出选中片段")
        self.export_btn.setEnabled(False)
        self.export_btn.clicked.connect(self._export_clip)
        yellow_btn_style = "QPushButton { color: #000; padding: 8px 20px; font-size: 13px; background-color: #FFC107; } QPushButton:disabled { background-color: #555; color: #888; } QPushButton:hover { background-color: #FFA000; }"
        self.export_btn.setStyleSheet(yellow_btn_style)
        btn_layout.addWidget(self.export_btn)

        btn_layout.addSpacing(40)

        # 播放控制 - 绿色播放，红色停止
        self.play_green_style = "QPushButton { color: white; padding: 8px 15px; font-size: 13px; background-color: #4CAF50; min-width: 70px; } QPushButton:disabled { background-color: #555; color: #888; } QPushButton:hover { background-color: #388E3C; }"
        self.play_red_style = "QPushButton { color: white; padding: 8px 15px; font-size: 13px; background-color: #F44336; min-width: 70px; } QPushButton:disabled { background-color: #555; color: #888; } QPushButton:hover { background-color: #D32F2F; }"

        self.play_btn = QPushButton("播放")
        self.play_btn.clicked.connect(self._toggle_play)
        self.play_btn.setEnabled(False)
        self.play_btn.setStyleSheet(self.play_green_style)
        btn_layout.addWidget(self.play_btn)

        self.play_range_btn = QPushButton("区间播放")
        self.play_range_btn.clicked.connect(self._toggle_play_range)
        self.play_range_btn.setEnabled(False)
        self.play_range_btn.setStyleSheet(self.play_green_style)
        btn_layout.addWidget(self.play_range_btn)

        btn_layout.addSpacing(20)

        # 播放速度
        speed_label = QLabel("速度:")
        speed_label.setStyleSheet("font-size: 14px; font-weight: bold;")
        btn_layout.addWidget(speed_label)
        self.speed_normal_style = "QPushButton { font-size: 14px; padding: 5px 10px; } QPushButton:disabled { background-color: #555; color: #888; }"
        self.speed_selected_style = "QPushButton { font-size: 14px; padding: 5px 10px; font-weight: bold; color: white; background-color: #1976D2; } QPushButton:disabled { background-color: #555; color: #888; }"
        self.speed_buttons = []
        self.selected_speed_btn = None
        speeds = [("×0.5", 0.5), ("×1", 1.0), ("×2", 2.0), ("×4", 4.0)]
        for name, speed in speeds:
            btn = QPushButton(name)
            btn.setEnabled(False)
            btn.setStyleSheet(self.speed_normal_style)
            btn.clicked.connect(lambda checked, s=speed, b=btn: self._set_speed(s, b))
            btn_layout.addWidget(btn)
            self.speed_buttons.append(btn)
            if speed == 1.0:
                self.selected_speed_btn = btn

        btn_layout.addStretch()
        layout.addLayout(btn_layout)

        # 快速时长按钮 + 宽高比选择
        duration_group = QGroupBox("快速设置时长 / 宽高比")
        duration_layout = QHBoxLayout(duration_group)

        durations = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60]
        self.duration_buttons = []
        self.duration_normal_style = "QPushButton { font-size: 14px; } QPushButton:disabled { background-color: #BDBDBD; color: #757575; }"
        self.duration_selected_style = "QPushButton { font-size: 14px; font-weight: bold; color: white; background-color: #1976D2; }"
        self.selected_duration_btn = None
        for sec in durations:
            btn = QPushButton(f"{sec}秒")
            btn.setEnabled(False)
            btn.setStyleSheet(self.duration_normal_style)
            btn.clicked.connect(lambda checked, s=sec, b=btn: self._set_duration(s, b))
            btn.setFixedWidth(55)
            btn.setFixedHeight(30)
            duration_layout.addWidget(btn)
            self.duration_buttons.append(btn)

        duration_layout.addSpacing(30)

        # 宽高比选择
        aspect_label = QLabel("宽高比:")
        aspect_label.setStyleSheet("font-size: 14px; font-weight: bold;")
        duration_layout.addWidget(aspect_label)

        self.aspect_ratios = [
            ("自由", None),
            ("1:1", 1.0),
            ("4:3", 4/3),
            ("3:4", 3/4),
            ("16:9", 16/9),
            ("9:16", 9/16),
        ]

        self.aspect_normal_style = "QPushButton { font-size: 14px; padding: 5px 10px; } QPushButton:disabled { background-color: #555; color: #888; }"
        self.aspect_selected_style = "QPushButton { font-size: 14px; padding: 5px 10px; font-weight: bold; color: white; background-color: #1976D2; } QPushButton:disabled { background-color: #555; color: #888; }"
        self.aspect_buttons = []
        self.selected_aspect_btn = None
        for i, (name, ratio) in enumerate(self.aspect_ratios):
            btn = QPushButton(name)
            btn.setEnabled(False)
            btn.setStyleSheet(self.aspect_normal_style)
            btn.clicked.connect(lambda checked, r=ratio, b=btn: self._set_aspect_ratio(r, b))
            duration_layout.addWidget(btn)
            self.aspect_buttons.append(btn)
            if i == 0:
                self.selected_aspect_btn = btn

        duration_layout.addStretch()

        layout.addWidget(duration_group)

        # 底部状态栏
        self.status_label = QLabel("就绪")
        self.status_label.setStyleSheet("color: #888; padding: 5px; border-top: 1px solid #444;")
        layout.addWidget(self.status_label)

    def _select_folder(self):
        """选择文件夹"""
        folder = QFileDialog.getExistingDirectory(self, "选择文件夹", str(self.current_folder))
        if folder:
            self.current_folder = Path(folder)
            self.folder_label.setText(self.current_folder.name)
            # 更新生成文件夹路径
            self.generated_folder = self.current_folder / "生成"
            if not self.generated_folder.exists():
                self.generated_folder.mkdir(exist_ok=True)
            self._refresh_file_list()

    def _refresh_file_list(self):
        """刷新文件列表"""
        video_extensions = {'.mp4', '.avi', '.mkv', '.mov', '.wmv', '.flv', '.webm'}

        # 刷新源视频列表
        self.file_list.clear()
        if self.current_folder.exists():
            files = sorted(self.current_folder.iterdir(), key=lambda x: x.name.lower())
            for f in files:
                if f.is_file() and f.suffix.lower() in video_extensions:
                    item = QListWidgetItem(f.name)
                    item.setData(Qt.UserRole, str(f))
                    self.file_list.addItem(item)

        # 刷新生成视频列表（按修改时间从新到旧排序）
        self.generated_list.clear()
        if self.generated_folder.exists():
            files = sorted(self.generated_folder.iterdir(),
                          key=lambda x: x.stat().st_mtime, reverse=True)
            for f in files:
                if f.is_file() and f.suffix.lower() in video_extensions:
                    item = QListWidgetItem(f.name)
                    item.setData(Qt.UserRole, str(f))
                    self.generated_list.addItem(item)

    def _on_generated_selected(self, item):
        """生成视频列表点击"""
        path = item.data(Qt.UserRole)
        if path:
            self.file_list.clearSelection()
            self._load_video(path)

    def _on_generated_right_click(self, pos):
        """生成视频列表右键 - 在文件浏览器中打开"""
        item = self.generated_list.itemAt(pos)
        if item:
            path = item.data(Qt.UserRole)
            if path:
                self._open_in_explorer(path)

    def _open_in_explorer(self, file_path):
        """在文件浏览器中打开文件所在位置"""
        import subprocess
        import platform
        if platform.system() == "Windows":
            subprocess.run(['explorer', '/select,', file_path])
        elif platform.system() == "Darwin":  # macOS
            subprocess.run(['open', '-R', file_path])
        else:  # Linux
            subprocess.run(['xdg-open', str(Path(file_path).parent)])

    def _on_file_selected(self, item):
        """文件列表点击"""
        path = item.data(Qt.UserRole)
        if path:
            self.generated_list.clearSelection()
            self._load_video(path)

    def _load_video(self, path):
        """加载视频文件"""
        # 停止播放
        if self.is_playing:
            self._stop_play()

        if self.cap:
            self.cap.release()

        self.cap = cv2.VideoCapture(path)
        if not self.cap.isOpened():
            self.status_label.setText(f"错误: 无法打开视频 {Path(path).name}")
            self.status_label.setStyleSheet("color: #F44336; padding: 5px; border-top: 1px solid #444;")
            return

        self.video_path = path
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # 计算视频总时长
        total_seconds = self.total_frames / self.fps
        hours = int(total_seconds // 3600)
        minutes = int((total_seconds % 3600) // 60)
        seconds = int(total_seconds % 60)
        duration_str = f"{hours:02d}:{minutes:02d}:{seconds:02d}"

        self.file_label.setText(
            f"{Path(path).name} ({self.width}x{self.height}, {self.fps:.1f}fps, {duration_str}, {self.total_frames}帧)"
        )
        self.file_label.setStyleSheet("color: white;")

        self.range_slider.setRange(0, self.total_frames - 1)

        # 启用按钮
        self.set_start_btn.setEnabled(True)
        self.set_end_btn.setEnabled(True)
        self.export_btn.setEnabled(True)
        self.play_btn.setEnabled(True)
        self.play_range_btn.setEnabled(True)
        for btn in self.duration_buttons:
            btn.setEnabled(True)
        for btn in self.speed_buttons:
            btn.setEnabled(True)
        for btn in self.aspect_buttons:
            btn.setEnabled(True)

        # 设置视频尺寸
        self.preview_label.setVideoSize(self.width, self.height)

        self._preview_frame(0)
        self._update_time_labels()

        # 重置裁剪区域（在预览帧之后，确保图片尺寸已更新）
        self.preview_label.resetCrop()
        # 重置宽高比为自由
        if self.selected_aspect_btn:
            self.selected_aspect_btn.setStyleSheet(self.aspect_normal_style)
        self.aspect_buttons[0].setStyleSheet(self.aspect_selected_style)
        self.selected_aspect_btn = self.aspect_buttons[0]
        self.preview_label.setAspectRatio(None)

        self.status_label.setText(f"已加载: {Path(path).name}")
        self.status_label.setStyleSheet("color: #4CAF50; padding: 5px; border-top: 1px solid #444;")

    def _open_video(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "选择视频文件", str(self.current_folder),
            "视频文件 (*.mp4 *.avi *.mkv *.mov *.wmv);;所有文件 (*)"
        )
        if path:
            self._load_video(path)

    def _preview_frame(self, frame_idx):
        if not self.cap:
            return

        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = self.cap.read()

        if not ret:
            return

        h, w = frame.shape[:2]
        max_h = self.preview_label.height() - 10
        max_w = self.preview_label.width() - 10

        scale = min(max_w / w, max_h / h)
        new_w, new_h = int(w * scale), int(h * scale)

        if new_w > 0 and new_h > 0:
            resized = cv2.resize(frame, (new_w, new_h))
            rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            qimg = QImage(rgb.data, new_w, new_h, new_w * 3, QImage.Format_RGB888)
            self.preview_label.setPixmap(QPixmap.fromImage(qimg))

    def _on_start_changed(self, val):
        self._preview_frame(val)
        self._update_time_labels()
        self._reset_duration_selection()

    def _on_end_changed(self, val):
        self._preview_frame(val)
        self._update_time_labels()
        self._reset_duration_selection()

    def _on_preview_changed(self, val):
        self._preview_frame(val)
        self._update_time_labels()

    def _on_crop_changed(self, crop):
        x, y, w, h = crop
        self.crop_info_label.setText(
            f"裁剪: X={x*100:.0f}% Y={y*100:.0f}% W={w*100:.0f}% H={h*100:.0f}%"
        )

    def _set_aspect_ratio(self, ratio, btn=None):
        """设置宽高比"""
        self.preview_label.setAspectRatio(ratio)
        # 更新按钮样式
        if btn:
            if self.selected_aspect_btn:
                self.selected_aspect_btn.setStyleSheet(self.aspect_normal_style)
            btn.setStyleSheet(self.aspect_selected_style)
            self.selected_aspect_btn = btn

    def _reset_crop(self):
        """重置裁剪区域"""
        self.preview_label.resetCrop()
        # 重新应用当前宽高比
        if self.selected_aspect_btn:
            idx = self.aspect_buttons.index(self.selected_aspect_btn)
            current_ratio = self.aspect_ratios[idx][1]
            if current_ratio is not None:
                self.preview_label.setAspectRatio(current_ratio)

    def _reset_duration_selection(self):
        """重置时长按钮选中状态"""
        if self.selected_duration_btn:
            self.selected_duration_btn.setStyleSheet(self.duration_normal_style)
            self.selected_duration_btn = None

    def _set_preview_as_start(self):
        preview = self.range_slider.preview()
        # 如果预览位置在终点之后，把终点也设为预览位置
        if preview > self.range_slider.end():
            self.range_slider.setEnd(preview)
        self.range_slider.setStart(preview)
        self._reset_duration_selection()

    def _set_preview_as_end(self):
        preview = self.range_slider.preview()
        # 如果预览位置在起点之前，把起点也设为预览位置
        if preview < self.range_slider.start():
            self.range_slider.setStart(preview)
        self.range_slider.setEnd(preview)
        self._reset_duration_selection()

    def _set_duration(self, seconds, btn=None):
        """设置时长（从起点开始）"""
        frames = int(seconds * self.fps)
        new_end = self.range_slider.start() + frames
        new_end = min(new_end, self.total_frames - 1)
        self.range_slider.setEnd(new_end)

        # 更新按钮样式
        if btn:
            if self.selected_duration_btn:
                self.selected_duration_btn.setStyleSheet(self.duration_normal_style)
            btn.setStyleSheet(self.duration_selected_style)
            self.selected_duration_btn = btn

    def _toggle_play(self):
        """切换播放/暂停"""
        if not self.cap:
            return
        if self.is_playing:
            self._stop_play()
        else:
            self.play_range_only = False
            # 定位到当前预览位置
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.range_slider.preview())
            self._start_play()

    def _toggle_play_range(self):
        """切换播放区间/暂停"""
        if not self.cap:
            return
        if self.is_playing:
            self._stop_play()
        else:
            self.play_range_only = True
            # 定位到起点
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.range_slider.start())
            self.range_slider._preview = self.range_slider.start()
            self._start_play()

    def _start_play(self):
        """开始播放"""
        self.is_playing = True
        self.play_btn.setText("暂停")
        self.play_btn.setStyleSheet(self.play_red_style)
        self.play_range_btn.setText("停止")
        self.play_range_btn.setStyleSheet(self.play_red_style)
        self._update_play_interval()

    def _update_play_interval(self):
        """更新播放定时器间隔和跳帧数"""
        # 初始跳帧数 = 播放速度
        if self.play_speed >= 2:
            self.frame_skip = int(self.play_speed)
        else:
            self.frame_skip = 1  # 0.5x和1x每帧都显示

        # 重置时间记录
        self.last_play_time = time.time()

        # 计算定时器间隔（考虑跳帧后的实际帧率）
        effective_fps = (self.fps * self.play_speed) / self.frame_skip
        interval = int(1000 / effective_fps)
        self.play_timer.start(max(10, interval))  # 最小10ms

    def _stop_play(self):
        """停止播放"""
        self.play_timer.stop()
        self.is_playing = False
        self.play_btn.setText("播放")
        self.play_btn.setStyleSheet(self.play_green_style)
        self.play_range_btn.setText("区间播放")
        self.play_range_btn.setStyleSheet(self.play_green_style)

    def _set_speed(self, speed, btn=None):
        """设置播放速度"""
        self.play_speed = speed
        if self.is_playing:
            self._update_play_interval()
        # 更新按钮样式
        if btn:
            if self.selected_speed_btn:
                self.selected_speed_btn.setStyleSheet(self.speed_normal_style)
            btn.setStyleSheet(self.speed_selected_style)
            self.selected_speed_btn = btn

    def _on_play_timer(self):
        """播放定时器回调 - 顺序读取帧，支持动态跳帧加速"""
        current = self.range_slider._preview
        now = time.time()

        # 动态调整跳帧数（仅4x速度）
        if self.play_speed >= 4 and self.last_play_time > 0:
            actual_interval = now - self.last_play_time
            # 期望间隔：跳过frame_skip帧应该花费的真实时间
            expected_interval = self.frame_skip / (self.fps * self.play_speed)

            # 如果实际时间超过期望的1.2倍，说明跟不上，增加跳帧数
            if actual_interval > expected_interval * 1.2:
                # 根据延迟比例增加跳帧，最大不超过speed*2
                self.frame_skip = min(int(self.frame_skip * 1.5), int(self.play_speed * 2))

        self.last_play_time = now

        if self.play_range_only:
            # 区间播放模式 - 到达终点后停止
            if current >= self.range_slider.end():
                self._stop_play()
                return
        else:
            # 普通播放模式
            if current >= self.total_frames - 1:
                self._stop_play()
                return

        # 根据 frame_skip 跳帧读取，只显示最后一帧
        frame = None
        for i in range(self.frame_skip):
            ret, frame = self.cap.read()
            if not ret:
                self._stop_play()
                return
            current += 1

            # 检查边界
            if self.play_range_only:
                if current >= self.range_slider.end():
                    break
            else:
                if current >= self.total_frames - 1:
                    break

        # 更新预览位置
        self.range_slider._preview = current
        self.range_slider.update()
        self._update_time_labels()

        # 显示帧
        self._display_frame(frame)

    def _display_frame(self, frame):
        """显示帧到预览区域"""
        h, w = frame.shape[:2]
        max_h = self.preview_label.height() - 10
        max_w = self.preview_label.width() - 10

        scale = min(max_w / w, max_h / h)
        new_w, new_h = int(w * scale), int(h * scale)

        if new_w > 0 and new_h > 0:
            resized = cv2.resize(frame, (new_w, new_h))
            rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            qimg = QImage(rgb.data, new_w, new_h, new_w * 3, QImage.Format_RGB888)
            self.preview_label.setPixmap(QPixmap.fromImage(qimg))

    def _update_time_labels(self):
        start_frame = self.range_slider.start()
        end_frame = self.range_slider.end()
        preview_frame = self.range_slider.preview()

        self.start_time_label.setText(f"起点: {self._frame_to_time(start_frame)}")
        self.preview_time_label.setText(f"预览: {self._frame_to_time(preview_frame)}")
        self.end_time_label.setText(f"终点: {self._frame_to_time(end_frame)}")
        self.duration_label.setText(f"时长: {self._frame_to_time(end_frame - start_frame)}")

    def _frame_to_time(self, frame):
        if self.fps <= 0:
            return "00:00:00"
        seconds = frame / self.fps
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        s = int(seconds % 60)
        return f"{h:02d}:{m:02d}:{s:02d}"

    def _export_clip(self):
        if not self.cap or not self.video_path:
            return

        start_frame = self.range_slider.start()
        end_frame = self.range_slider.end()

        if end_frame <= start_frame:
            self.status_label.setText("错误: 请选择有效的剪切范围")
            self.status_label.setStyleSheet("color: #F44336; padding: 5px; border-top: 1px solid #444;")
            return

        # 获取裁剪区域
        crop_x, crop_y, crop_w, crop_h = self.preview_label.getCrop()

        # 计算实际像素坐标
        cx = int(crop_x * self.width)
        cy = int(crop_y * self.height)
        cw = int(crop_w * self.width)
        ch = int(crop_h * self.height)

        # 确保尺寸为偶数（某些编码器需要）
        cw = cw - (cw % 2)
        ch = ch - (ch % 2)

        total = end_frame - start_frame
        duration_sec = round(total / self.fps)

        # 用时间作为文件名，加上时长，保存到生成文件夹
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = str(self.generated_folder / f"{timestamp}_{duration_sec}秒.mp4")

        # 创建进度对话框（模态，阻止其他操作）
        progress = QProgressDialog("正在导出视频...", "停止", 0, total, self)
        progress.setWindowTitle("导出中")
        progress.setWindowModality(Qt.WindowModal)
        progress.setMinimumDuration(0)
        progress.setValue(0)
        progress.show()

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(save_path, fourcc, self.fps, (cw, ch))

        self.cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        canceled = False
        for i in range(total):
            if progress.wasCanceled():
                canceled = True
                break

            ret, frame = self.cap.read()
            if not ret:
                break

            # 裁剪
            cropped = frame[cy:cy+ch, cx:cx+cw]
            writer.write(cropped)

            # 更新进度
            progress.setValue(i + 1)
            progress.setLabelText(f"正在导出... {(i+1)/total*100:.0f}%\n{i+1}/{total} 帧")
            QApplication.processEvents()

        writer.release()
        progress.close()

        if canceled:
            # 删除未完成的文件
            try:
                Path(save_path).unlink()
            except:
                pass
            self.status_label.setText("导出已取消")
            self.status_label.setStyleSheet("color: #FFC107; padding: 5px; border-top: 1px solid #444;")
            return

        # 刷新文件列表
        self._refresh_file_list()

        # 显示完成信息
        duration = self._frame_to_time(total)
        self.status_label.setText(
            f"导出完成: {Path(save_path).name} | "
            f"时长: {duration} | 分辨率: {cw}x{ch}"
        )
        self.status_label.setStyleSheet("color: #4CAF50; padding: 5px; border-top: 1px solid #444;")

    def closeEvent(self, event):
        if self.is_playing:
            self._stop_play()
        if self.cap:
            self.cap.release()
        event.accept()


def main():
    app = QApplication(sys.argv)
    app.setStyle('Fusion')

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

    window = VideoCutWindow()
    window.showMaximized()

    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
