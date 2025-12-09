"""
测试YOLO检测器 - 调试用
"""
import cv2
import numpy as np
from pathlib import Path

def test_with_image(image_path=None):
    """用图片测试检测器"""

    model_path = Path(__file__).parent / "models" / "yolov8n.onnx"
    print(f"模型路径: {model_path}")
    print(f"模型存在: {model_path.exists()}")
    print(f"模型大小: {model_path.stat().st_size / 1024 / 1024:.2f} MB")

    # 加载模型
    net = cv2.dnn.readNetFromONNX(str(model_path))
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

    # 准备测试图像
    if image_path and Path(image_path).exists():
        image = cv2.imread(image_path)
        print(f"使用图片: {image_path}")
    else:
        # 创建一个有简单图案的测试图像
        image = np.full((480, 640, 3), 200, dtype=np.uint8)
        # 画一个简单的人形轮廓
        cv2.rectangle(image, (280, 100), (360, 400), (100, 100, 100), -1)  # 身体
        cv2.circle(image, (320, 70), 40, (100, 100, 100), -1)  # 头
        print("使用生成的测试图像")

    print(f"图像尺寸: {image.shape}")

    # 预处理
    input_size = 640
    h, w = image.shape[:2]
    scale = min(input_size / w, input_size / h)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(image, (new_w, new_h))

    pad_w = (input_size - new_w) // 2
    pad_h = (input_size - new_h) // 2
    padded = np.full((input_size, input_size, 3), 114, dtype=np.uint8)
    padded[pad_h:pad_h+new_h, pad_w:pad_w+new_w] = resized

    blob = cv2.dnn.blobFromImage(padded, 1/255.0, (input_size, input_size), swapRB=True, crop=False)
    print(f"Blob shape: {blob.shape}")

    # 推理
    net.setInput(blob)
    outputs = net.forward()
    print(f"Output shape: {outputs.shape}")

    # 分析输出
    if len(outputs.shape) == 3:
        # YOLOv8格式: [1, 84, 8400]
        print(f"检测到 YOLOv8 格式输出")
        data = outputs[0].T  # [8400, 84]
        print(f"转置后: {data.shape}")

        # 查看所有类别的最大置信度
        boxes = data[:, :4]
        class_scores = data[:, 4:]  # 80个类别

        print(f"\n各类别最大置信度 (前10个类别):")
        for i in range(min(10, class_scores.shape[1])):
            max_score = class_scores[:, i].max()
            print(f"  类别 {i}: {max_score:.4f}")

        # 人物类别 (class 0) 的分析
        person_scores = class_scores[:, 0]
        print(f"\n人物检测分析:")
        print(f"  最大置信度: {person_scores.max():.4f}")
        print(f"  置信度 > 0.1 的数量: {(person_scores > 0.1).sum()}")
        print(f"  置信度 > 0.3 的数量: {(person_scores > 0.3).sum()}")
        print(f"  置信度 > 0.5 的数量: {(person_scores > 0.5).sum()}")

        # 找出最高分的检测
        best_idx = person_scores.argmax()
        print(f"\n最佳检测:")
        print(f"  索引: {best_idx}")
        print(f"  置信度: {person_scores[best_idx]:.4f}")
        print(f"  边界框 (cx, cy, w, h): {boxes[best_idx]}")

    else:
        print(f"未知输出格式: {outputs.shape}")


def test_with_video_frame(video_path):
    """从视频中提取一帧测试"""
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()

    if ret:
        # 保存帧用于测试
        test_frame_path = Path(__file__).parent / "test_frame.jpg"
        cv2.imwrite(str(test_frame_path), frame)
        print(f"已保存测试帧: {test_frame_path}")
        test_with_image(str(test_frame_path))
    else:
        print("无法读取视频")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        path = sys.argv[1]
        if path.endswith(('.mp4', '.avi', '.mkv', '.mov')):
            test_with_video_frame(path)
        else:
            test_with_image(path)
    else:
        test_with_image()
