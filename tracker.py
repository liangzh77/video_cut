"""
多人跟踪器 - 使用OpenCV内置跟踪算法
"""

import cv2
import numpy as np
from enum import Enum
from typing import Optional, Tuple, List, Dict


class TrackerType(Enum):
    """可用的跟踪器类型"""
    KCF = "KCF"
    CSRT = "CSRT"
    MOSSE = "MOSSE"


def create_tracker(tracker_type: TrackerType):
    """创建OpenCV跟踪器实例 (兼容不同版本)"""
    if hasattr(cv2, 'legacy'):
        if tracker_type == TrackerType.KCF:
            return cv2.legacy.TrackerKCF_create()
        elif tracker_type == TrackerType.CSRT:
            return cv2.legacy.TrackerCSRT_create()
        elif tracker_type == TrackerType.MOSSE:
            return cv2.legacy.TrackerMOSSE_create()
    else:
        if tracker_type == TrackerType.KCF:
            return cv2.TrackerKCF_create()
        elif tracker_type == TrackerType.CSRT:
            return cv2.TrackerCSRT_create()
        elif tracker_type == TrackerType.MOSSE:
            return cv2.TrackerMOSSE_create()

    if hasattr(cv2, 'legacy'):
        return cv2.legacy.TrackerCSRT_create()
    return cv2.TrackerCSRT_create()


def calc_iou(box1: Tuple, box2: Tuple) -> float:
    """计算两个框的IoU"""
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2

    xi1 = max(x1_1, x1_2)
    yi1 = max(y1_1, y1_2)
    xi2 = min(x2_1, x2_2)
    yi2 = min(y2_1, y2_2)

    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)

    box1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
    box2_area = (x2_2 - x1_2) * (y2_2 - y1_2)

    union_area = box1_area + box2_area - inter_area

    if union_area == 0:
        return 0
    return inter_area / union_area


class TrackedPerson:
    """单个被跟踪的人"""

    def __init__(self, person_id: int, bbox: Tuple, frame: np.ndarray,
                 tracker_type: TrackerType):
        self.id = person_id
        self.bbox = bbox  # (x1, y1, x2, y2)
        self.confidence = 1.0
        self.frames_since_detection = 0
        self.lost_frames = 0
        self.is_active = True

        # 创建跟踪器
        self.tracker = create_tracker(tracker_type)
        x1, y1, x2, y2 = bbox
        cv_bbox = (x1, y1, x2 - x1, y2 - y1)
        self.tracker.init(frame, cv_bbox)

    def update_with_detection(self, bbox: Tuple, frame: np.ndarray,
                               tracker_type: TrackerType):
        """用新的检测结果更新"""
        self.bbox = bbox
        self.confidence = 1.0
        self.frames_since_detection = 0
        self.lost_frames = 0

        # 重新初始化跟踪器
        self.tracker = create_tracker(tracker_type)
        x1, y1, x2, y2 = bbox
        cv_bbox = (x1, y1, x2 - x1, y2 - y1)
        self.tracker.init(frame, cv_bbox)

    def update_with_tracker(self, frame: np.ndarray) -> bool:
        """用跟踪器更新"""
        success, cv_bbox = self.tracker.update(frame)

        if success:
            x, y, w, h = [int(v) for v in cv_bbox]
            self.bbox = (x, y, x + w, y + h)
            self.frames_since_detection += 1
            self.confidence = max(0.3, 1.0 - self.frames_since_detection * 0.02)
            return True
        else:
            self.lost_frames += 1
            if self.lost_frames > 10:
                self.is_active = False
            return False


class MultiPersonTracker:
    """
    多人跟踪器

    策略:
    1. 使用YOLO检测所有人
    2. 为每个人分配唯一ID
    3. 使用IoU匹配新检测与已有轨迹
    4. 在检测间隔使用轻量跟踪器
    """

    def __init__(self, detector, tracker_type: TrackerType = TrackerType.CSRT,
                 redetect_interval: int = 30, iou_threshold: float = 0.3):
        self.detector = detector
        self.tracker_type = tracker_type
        self.redetect_interval = redetect_interval
        self.iou_threshold = iou_threshold

        self.tracked_persons: Dict[int, TrackedPerson] = {}
        self.next_id = 1
        self.frame_count = 0

    def process_frame(self, frame: np.ndarray) -> List[Tuple[int, Tuple, float, str]]:
        """
        处理一帧

        Returns:
            列表 [(person_id, bbox, confidence, method), ...]
            method: "yolo" | "tracker"
        """
        self.frame_count += 1
        results = []

        # 判断是否需要YOLO检测
        need_yolo = (
            self.frame_count == 1 or
            self.frame_count % self.redetect_interval == 0 or
            len(self.tracked_persons) == 0
        )

        if need_yolo:
            results = self._process_with_yolo(frame)
        else:
            results = self._process_with_trackers(frame)

        # 清理不活跃的轨迹
        self._cleanup_inactive()

        return results

    def _process_with_yolo(self, frame: np.ndarray) -> List:
        """使用YOLO检测并匹配"""
        detections = self.detector.detect(frame)
        results = []

        if not detections:
            # 没有检测到，用跟踪器更新现有轨迹
            return self._process_with_trackers(frame)

        # 匹配检测与现有轨迹
        matched_track_ids = set()
        matched_det_indices = set()

        # 计算所有IoU
        if self.tracked_persons:
            for det_idx, det in enumerate(detections):
                det_bbox = det[:4]
                best_iou = 0
                best_track_id = None

                for track_id, person in self.tracked_persons.items():
                    if track_id in matched_track_ids:
                        continue
                    iou = calc_iou(det_bbox, person.bbox)
                    if iou > best_iou and iou > self.iou_threshold:
                        best_iou = iou
                        best_track_id = track_id

                if best_track_id is not None:
                    # 匹配成功，更新轨迹
                    self.tracked_persons[best_track_id].update_with_detection(
                        det_bbox, frame, self.tracker_type
                    )
                    matched_track_ids.add(best_track_id)
                    matched_det_indices.add(det_idx)
                    results.append((
                        best_track_id,
                        det_bbox,
                        det[4],
                        "yolo"
                    ))

        # 为未匹配的检测创建新轨迹
        for det_idx, det in enumerate(detections):
            if det_idx not in matched_det_indices:
                det_bbox = det[:4]
                new_id = self.next_id
                self.next_id += 1

                self.tracked_persons[new_id] = TrackedPerson(
                    new_id, det_bbox, frame, self.tracker_type
                )
                results.append((new_id, det_bbox, det[4], "yolo"))

        # 用跟踪器更新未匹配的现有轨迹
        for track_id, person in self.tracked_persons.items():
            if track_id not in matched_track_ids:
                if person.update_with_tracker(frame):
                    results.append((
                        track_id,
                        person.bbox,
                        person.confidence,
                        "tracker"
                    ))

        return results

    def _process_with_trackers(self, frame: np.ndarray) -> List:
        """只使用跟踪器更新"""
        results = []

        for track_id, person in self.tracked_persons.items():
            if person.is_active:
                if person.update_with_tracker(frame):
                    results.append((
                        track_id,
                        person.bbox,
                        person.confidence,
                        "tracker"
                    ))

        return results

    def _cleanup_inactive(self):
        """清理不活跃的轨迹"""
        inactive_ids = [
            tid for tid, p in self.tracked_persons.items()
            if not p.is_active
        ]
        for tid in inactive_ids:
            del self.tracked_persons[tid]

    def reset(self):
        """重置所有状态"""
        self.tracked_persons.clear()
        self.next_id = 1
        self.frame_count = 0


# 保留旧接口兼容
class LightweightTracker:
    def __init__(self, tracker_type: TrackerType = TrackerType.CSRT):
        self.tracker_type = tracker_type
        self.tracker = None
        self.bbox = None
        self.is_initialized = False
        self.frames_since_init = 0
        self.confidence = 1.0

    def _create_tracker(self):
        return create_tracker(self.tracker_type)

    def init(self, frame: np.ndarray, bbox: Tuple[int, int, int, int]) -> bool:
        x1, y1, x2, y2 = bbox
        cv_bbox = (x1, y1, x2 - x1, y2 - y1)
        if cv_bbox[2] <= 0 or cv_bbox[3] <= 0:
            return False
        self.tracker = self._create_tracker()
        success = self.tracker.init(frame, cv_bbox)
        if success:
            self.bbox = bbox
            self.is_initialized = True
            self.frames_since_init = 0
            self.confidence = 1.0
        return success

    def update(self, frame: np.ndarray):
        if not self.is_initialized or self.tracker is None:
            return False, None, 0.0
        success, cv_bbox = self.tracker.update(frame)
        if success:
            x, y, w, h = [int(v) for v in cv_bbox]
            self.bbox = (x, y, x + w, y + h)
            self.frames_since_init += 1
            self.confidence = max(0.3, 1.0 - self.frames_since_init * 0.02)
            return True, self.bbox, self.confidence
        else:
            self.is_initialized = False
            return False, None, 0.0

    def reset(self):
        self.tracker = None
        self.bbox = None
        self.is_initialized = False
        self.frames_since_init = 0
        self.confidence = 1.0

    def should_redetect(self, redetect_interval: int = 30) -> bool:
        if not self.is_initialized:
            return True
        if self.confidence < 0.5:
            return True
        if self.frames_since_init >= redetect_interval:
            return True
        return False

    def get_bbox(self):
        return self.bbox if self.is_initialized else None


class HybridTracker:
    """兼容旧接口"""
    def __init__(self, detector, tracker_type: TrackerType = TrackerType.CSRT,
                 redetect_interval: int = 30):
        self.multi_tracker = MultiPersonTracker(
            detector, tracker_type, redetect_interval
        )

    def process_frame(self, frame):
        results = self.multi_tracker.process_frame(frame)
        if results:
            # 返回第一个人的结果（兼容旧接口）
            pid, bbox, conf, method = results[0]
            return bbox, method, conf
        return None, "none", 0.0

    def reset(self):
        self.multi_tracker.reset()
