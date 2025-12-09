"""
视频处理器 - 多人跟踪版本
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Callable, Optional, List, Tuple
import time
import random

from yolo_detector import YOLODetector
from tracker import MultiPersonTracker, TrackerType


# 为每个ID分配固定颜色
def get_color_for_id(person_id: int) -> Tuple[int, int, int]:
    """根据ID生成固定颜色"""
    random.seed(person_id * 100)
    return (
        random.randint(50, 255),
        random.randint(50, 255),
        random.randint(50, 255)
    )


class VideoProcessor:
    """视频处理器 - 多人跟踪"""

    def __init__(self, detector: YOLODetector = None,
                 tracker_type: TrackerType = TrackerType.CSRT,
                 redetect_interval: int = 30):
        if detector is None:
            detector = YOLODetector()

        self.multi_tracker = MultiPersonTracker(
            detector=detector,
            tracker_type=tracker_type,
            redetect_interval=redetect_interval
        )

        self.is_processing = False
        self.should_stop = False

        self.stats = {
            "total_frames": 0,
            "yolo_frames": 0,
            "tracker_frames": 0,
            "total_persons": 0,
            "avg_fps": 0.0
        }

    def process_video(self, input_path: str, output_path: str = None,
                      progress_callback: Callable[[int, int, dict], None] = None,
                      preview_callback: Callable[[np.ndarray], None] = None,
                      skip_frames: int = 0) -> dict:
        """处理视频文件"""
        self.is_processing = True
        self.should_stop = False
        self.multi_tracker.reset()

        self.stats = {
            "total_frames": 0,
            "yolo_frames": 0,
            "tracker_frames": 0,
            "total_persons": 0,
            "avg_fps": 0.0
        }

        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise ValueError(f"无法打开视频文件: {input_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            effective_fps = fps / (skip_frames + 1)
            writer = cv2.VideoWriter(output_path, fourcc, effective_fps, (width, height))

        frame_count = 0
        start_time = time.time()

        try:
            while not self.should_stop:
                ret, frame = cap.read()
                if not ret:
                    break

                frame_count += 1

                if skip_frames > 0 and (frame_count - 1) % (skip_frames + 1) != 0:
                    continue

                self.stats["total_frames"] += 1

                # 多人跟踪
                results = self.multi_tracker.process_frame(frame)

                # 更新统计
                yolo_count = sum(1 for r in results if r[3] == "yolo")
                tracker_count = sum(1 for r in results if r[3] == "tracker")
                self.stats["yolo_frames"] += 1 if yolo_count > 0 else 0
                self.stats["tracker_frames"] += 1 if tracker_count > 0 and yolo_count == 0 else 0
                self.stats["total_persons"] = self.multi_tracker.next_id - 1

                # 绘制结果
                output_frame = self._draw_results(frame, results)

                if writer:
                    writer.write(output_frame)

                if preview_callback:
                    preview_callback(output_frame)

                if progress_callback:
                    elapsed = time.time() - start_time
                    if elapsed > 0:
                        self.stats["avg_fps"] = self.stats["total_frames"] / elapsed
                    progress_callback(frame_count, total_frames, self.stats.copy())

        finally:
            cap.release()
            if writer:
                writer.release()
            self.is_processing = False

        return self.stats

    def _draw_results(self, frame: np.ndarray,
                      results: List[Tuple[int, Tuple, float, str]]) -> np.ndarray:
        """绘制多人跟踪结果"""
        output = frame.copy()

        for person_id, bbox, confidence, method in results:
            x1, y1, x2, y2 = bbox

            # 根据ID获取颜色
            color = get_color_for_id(person_id)

            # 根据检测方法调整边框样式
            thickness = 3 if method == "yolo" else 2

            # 绘制边界框
            cv2.rectangle(output, (x1, y1), (x2, y2), color, thickness)

            # 绘制ID标签
            label = f"ID:{person_id}"
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)

            # 标签背景
            cv2.rectangle(output,
                          (x1, y1 - label_size[1] - 10),
                          (x1 + label_size[0] + 10, y1),
                          color, -1)

            # 标签文字
            cv2.putText(output, label, (x1 + 5, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

            # 置信度和方法（小字）
            info = f"{method} {confidence:.2f}"
            cv2.putText(output, info, (x1, y2 + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        # 统计信息
        active_count = len(results)
        stats_text = f"Tracking: {active_count} | Total IDs: {self.stats['total_persons']} | FPS: {self.stats['avg_fps']:.1f}"
        cv2.putText(output, stats_text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        return output

    def stop(self):
        self.should_stop = True


def process_video_cli(input_path: str, output_path: str = None,
                      redetect_interval: int = 30, skip_frames: int = 0):
    """命令行处理视频"""
    print(f"处理视频: {input_path}")
    print(f"重新检测间隔: {redetect_interval} 帧")

    processor = VideoProcessor(redetect_interval=redetect_interval)

    def progress_cb(current, total, stats):
        pct = current / total * 100
        print(f"\r进度: {pct:.1f}% | 跟踪人数: {stats['total_persons']} | FPS: {stats['avg_fps']:.1f}", end="")

    stats = processor.process_video(
        input_path=input_path,
        output_path=output_path,
        progress_callback=progress_cb,
        skip_frames=skip_frames
    )

    print("\n处理完成!")
    print(f"总帧数: {stats['total_frames']}")
    print(f"累计跟踪人数: {stats['total_persons']}")
    print(f"平均FPS: {stats['avg_fps']:.2f}")

    if output_path:
        print(f"输出已保存到: {output_path}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("用法: python video_processor.py <input_video> [output_video]")
        sys.exit(1)

    input_video = sys.argv[1]
    output_video = sys.argv[2] if len(sys.argv) > 2 else None

    process_video_cli(input_video, output_video)
