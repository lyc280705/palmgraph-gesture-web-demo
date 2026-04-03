# PalmGraph Gesture Web Demo

这个仓库只保留了可直接演示的最小内容：

- ONNX 模型
- Web 演示界面
- 运行演示所需的最少 Python 后端代码

未包含训练脚本、数据集、实验日志和其他研究文件。

## 目录说明

```text
.
├── README.md
├── requirements.txt
├── assets/
│   └── web_demo/
├── gesture_hgr/
├── models/
│   ├── model.int8.onnx
│   ├── model_meta.json
│   └── web_demo_bindings.json
└── scripts/
    └── demo_webui.py
```

## 环境要求

- macOS
- Python 3.10 或 3.11
- 默认使用本机摄像头

说明：

- 首次运行时，如果本地没有 MediaPipe 的 hand_landmarker.task，程序会自动下载到默认缓存位置。
- 系统级动作通过 macOS 原生能力触发；如果不需要控制动作，也可以只把它当作识别演示界面使用。

## 安装

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

## 启动演示

在仓库根目录执行：

```bash
python scripts/demo_webui.py
```

默认会：

- 加载 models/model.int8.onnx
- 读取 models/model_meta.json
- 读取 models/web_demo_bindings.json
- 启动本地网页界面 http://127.0.0.1:8000

## 常用参数

```bash
python scripts/demo_webui.py --camera 0 --width 1280 --height 720 --port 8000
```

如果你想显式指定模型路径：

```bash
python scripts/demo_webui.py \
  --onnx models/model.int8.onnx \
  --meta models/model_meta.json
```

## 仓库说明

这是一个发布版仓库，目标是方便直接演示，不包含训练与复现实验流程。