"""
下载 YOLOv8n ONNX 模型
"""
import urllib.request
import os
from pathlib import Path

MODEL_URLS = [
    # Hugging Face (带commit hash更稳定)
    "https://huggingface.co/Kalray/yolov8/resolve/main/yolov8n.onnx",
    "https://huggingface.co/SpotLab/YOLOv8Detection/resolve/3005c6751fb19cdeb6b10c066185908faf66a097/yolov8n.onnx",
    "https://huggingface.co/unity/inference-engine-yolo/resolve/ed7f4daf9263d0d31be1d60b9d67c8baea721d60/yolov8n.onnx",
]

def download():
    save_dir = Path(__file__).parent / "models"
    save_dir.mkdir(exist_ok=True)
    save_path = save_dir / "yolov8n.onnx"

    if save_path.exists():
        print(f"模型已存在: {save_path}")
        return

    for url in MODEL_URLS:
        try:
            print(f"尝试下载: {url}")
            urllib.request.urlretrieve(url, save_path, reporthook=progress)
            print(f"\n下载完成: {save_path}")
            return
        except Exception as e:
            print(f"\n失败: {e}")
            if save_path.exists():
                os.remove(save_path)

    print("\n所有下载源均失败!")
    print("请手动下载 yolov8n.onnx 并放入 models 文件夹")
    print("GitHub: https://github.com/ultralytics/assets/releases")

def progress(count, block_size, total_size):
    percent = count * block_size * 100 // total_size
    print(f"\r下载进度: {percent}%", end="", flush=True)

if __name__ == "__main__":
    download()
