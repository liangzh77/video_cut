# Video Cut Tool

基于 YOLOv8 的视频剪切工具，支持时间范围选择、区域裁剪和人物检测追踪。

## 功能特性

- **视频时间剪切**：通过三滑块控件（起点、终点、预览）精确选择视频片段
- **区域裁剪**：在预览画面上框选需要保留的区域
- **人物检测**：使用 YOLOv8 Nano 模型检测视频中的人物
- **目标追踪**：支持 KCF、CSRT、MOSSE 等多种追踪算法
- **实时预览**：拖动滑块即时预览视频帧
- **PyQt5 图形界面**：直观易用的操作界面

## 环境要求

- Python 3.8+
- OpenCV 4.8.0+
- NumPy 1.24.0+
- PyQt5 5.15.0+

## 安装

1. 克隆仓库：

```bash
git clone https://github.com/liangzh77/video_cut.git
cd video_cut
```

2. 安装依赖：

```bash
pip install -r requirements.txt
```

3. 下载模型文件：

从 Hugging Face 下载 YOLOv8n ONNX 模型：

**下载地址**：[yolov8n.onnx](https://huggingface.co/SpotLab/YOLOv8Detection/blob/3005c6751fb19cdeb6b10c066185908faf66a097/yolov8n.onnx)

将下载的 `yolov8n.onnx` 文件放入项目的 `models/` 目录：

```
video_cut/
├── models/
│   └── yolov8n.onnx    <-- 放在这里
├── main_cut.py
├── yolo_detector.py
└── ...
```

## 使用方法

运行主程序：

```bash
python main_cut.py
```

### 操作步骤

1. 点击 **打开视频** 选择要编辑的视频文件
2. 使用 **三滑块控件** 选择时间范围：
   - 绿色滑块：起点
   - 红色滑块：终点
   - 黄色三角：预览位置
3. 在预览区域 **框选裁剪区域**（可选）
4. 点击 **导出视频** 保存剪切后的视频

## 项目结构

```
video_cut/
├── main_cut.py         # 主程序入口，PyQt5 GUI
├── main_yolo.py        # YOLO 视频处理脚本
├── yolo_detector.py    # YOLOv8 检测器封装
├── tracker.py          # 多目标追踪器
├── video_processor.py  # 视频处理工具类
├── download_model.py   # 模型下载辅助脚本
├── requirements.txt    # 项目依赖
└── models/             # 模型文件目录
    └── yolov8n.onnx    # YOLOv8 Nano 模型
```

## License

MIT
