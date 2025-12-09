"""测试YOLO检测并可视化结果"""
import cv2
from pathlib import Path
from yolo_detector import YOLODetector

# 路径
image_path = Path(__file__).parent / "test.png"
output_path = Path(__file__).parent / "test_result.png"

# 加载检测器
print("加载检测器...")
detector = YOLODetector(conf_threshold=0.5)

# 读取图片
image = cv2.imread(str(image_path))
print(f"图片尺寸: {image.shape}")

# 检测
print("\n开始检测...")
results = detector.detect(image)
print(f"检测到 {len(results)} 个人")

# 打印检测结果
for i, (x1, y1, x2, y2, conf) in enumerate(results):
    print(f"  [{i}] x1={x1}, y1={y1}, x2={x2}, y2={y2}, conf={conf:.3f}")

# 绘制结果
output = image.copy()
for x1, y1, x2, y2, conf in results:
    cv2.rectangle(output, (x1, y1), (x2, y2), (0, 255, 0), 2)
    label = f"person {conf:.2f}"
    cv2.putText(output, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

# 保存结果
cv2.imwrite(str(output_path), output)
print(f"\n结果已保存到: {output_path}")
