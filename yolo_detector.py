"""
YOLO人物检测器 - 使用OpenCV DNN模块 (无需PyTorch/onnxruntime)
"""

import cv2
import numpy as np
from pathlib import Path


class YOLODetector:
    """YOLOv8 Nano 人物检测器 (OpenCV DNN版)"""

    PERSON_CLASS_ID = 0
    MODEL_NAME = "yolov8n.onnx"

    def __init__(self, model_path: str = None, conf_threshold: float = 0.5,
                 input_size: int = 640):
        """
        初始化检测器

        Args:
            model_path: ONNX模型路径
            conf_threshold: 置信度阈值
            input_size: 输入图像大小
        """
        self.conf_threshold = conf_threshold
        self.input_size = input_size

        # 确定模型路径
        if model_path is None:
            model_dir = Path(__file__).parent / "models"
            model_path = model_dir / self.MODEL_NAME

        if not Path(model_path).exists():
            raise FileNotFoundError(
                f"模型文件不存在: {model_path}\n"
                f"请手动下载 yolov8n.onnx 到 models 文件夹:\n"
                f"下载地址: https://github.com/ultralytics/assets/releases"
            )

        # 使用OpenCV DNN加载模型
        print(f"正在加载模型: {model_path}")
        self.net = cv2.dnn.readNetFromONNX(str(model_path))

        # 设置CPU后端
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

        print("YOLO检测器初始化完成")

    def detect(self, image: np.ndarray) -> list:
        """
        检测图像中的人物

        Args:
            image: BGR格式的OpenCV图像

        Returns:
            检测到的人物列表 [(x1, y1, x2, y2, confidence), ...]
        """
        h, w = image.shape[:2]

        # 预处理: letterbox + 归一化
        blob, scale, pad = self._preprocess(image)

        # 推理
        self.net.setInput(blob)
        outputs = self.net.forward()

        # 后处理
        detections = self._postprocess(outputs, scale, pad, w, h)

        return detections

    def _preprocess(self, image: np.ndarray):
        """预处理图像"""
        h, w = image.shape[:2]
        size = self.input_size

        # 计算缩放比例 (保持宽高比)
        scale = min(size / w, size / h)
        new_w, new_h = int(w * scale), int(h * scale)

        # 缩放
        resized = cv2.resize(image, (new_w, new_h))

        # 创建letterbox
        pad_w = (size - new_w) // 2
        pad_h = (size - new_h) // 2
        padded = np.full((size, size, 3), 114, dtype=np.uint8)
        padded[pad_h:pad_h+new_h, pad_w:pad_w+new_w] = resized

        # 转换为blob: BGR->RGB, HWC->NCHW, 归一化
        blob = cv2.dnn.blobFromImage(padded, 1/255.0, (size, size), swapRB=True, crop=False)

        return blob, scale, (pad_w, pad_h)

    def _postprocess(self, outputs: np.ndarray, scale: float, pad: tuple,
                     orig_w: int, orig_h: int) -> list:
        """后处理检测结果"""
        # YOLOv8输出: [1, 84, 8400] -> 转置为 [8400, 84]
        outputs = outputs[0].T

        # 提取边界框和人物类别置信度
        boxes = outputs[:, :4]  # cx, cy, w, h (归一化坐标 0-1)
        scores = outputs[:, 4 + self.PERSON_CLASS_ID]

        # 过滤低置信度
        mask = scores > self.conf_threshold
        boxes = boxes[mask]
        scores = scores[mask]

        if len(boxes) == 0:
            return []

        # 归一化坐标转像素坐标 (乘以input_size)
        boxes = boxes * self.input_size

        # 转换坐标: cx,cy,w,h -> x1,y1,x2,y2
        cx, cy, bw, bh = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        x1 = cx - bw / 2
        y1 = cy - bh / 2
        x2 = cx + bw / 2
        y2 = cy + bh / 2

        # 还原到原图坐标 (去除padding和缩放)
        pad_w, pad_h = pad
        x1 = (x1 - pad_w) / scale
        y1 = (y1 - pad_h) / scale
        x2 = (x2 - pad_w) / scale
        y2 = (y2 - pad_h) / scale

        # 裁剪到图像边界
        x1 = np.clip(x1, 0, orig_w)
        y1 = np.clip(y1, 0, orig_h)
        x2 = np.clip(x2, 0, orig_w)
        y2 = np.clip(y2, 0, orig_h)

        # NMS
        boxes_for_nms = np.stack([x1, y1, x2 - x1, y2 - y1], axis=1).tolist()
        indices = cv2.dnn.NMSBoxes(boxes_for_nms, scores.tolist(), self.conf_threshold, 0.45)

        results = []
        for i in indices:
            idx = i[0] if isinstance(i, (list, np.ndarray)) else i
            results.append((int(x1[idx]), int(y1[idx]), int(x2[idx]), int(y2[idx]), float(scores[idx])))

        results.sort(key=lambda x: x[4], reverse=True)
        return results

    def detect_largest_person(self, image: np.ndarray) -> tuple:
        """检测面积最大的人物"""
        results = self.detect(image)
        if not results:
            return None
        return max(results, key=lambda d: (d[2]-d[0]) * (d[3]-d[1]))


if __name__ == "__main__":
    detector = YOLODetector()
    test_img = np.zeros((480, 640, 3), dtype=np.uint8)
    results = detector.detect(test_img)
    print(f"检测到 {len(results)} 个人物")
