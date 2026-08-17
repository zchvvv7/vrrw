# 车载视觉道路障碍物与未知风险预警系统

## 1. 项目简介

本项目是一个基于前视视频的道路风险预警系统，通过障碍物检测判断当前是否存在行车风险。

当前系统版本：`vrrws_v1.0`。

## 2. 主流程

```text
输入视频 -> SegFormer道路分割 -> YOLO已知障碍物检测 -> 已知障碍物距离估计 -> Mask2Anomaly未知障碍物检测 -> 障碍物距离估计 -> 自车行驶走廊预测 -> 空间冲突、TTC 与风险评估 -> 右侧风险 UI、JSON 和日志输出
```

风险等级包括：

- `safe`：当前没有确认的空间冲突；
- `notice`：需要关注，但尚未达到警告条件；
- `warning`：存在较明显的距离或 TTC 风险；
- `danger`：障碍物距离或 TTC 已达到危险阈值。

## 3. 已实现功能

- 本地视频读取、跳帧处理和结果视频写入
- 可行驶区域分割和图像质量降级判断（SegFormer）
- 锥桶、护栏、坑洞、车辆等已知障碍物检测（YOLO）
- 像素级未知异常检测（Mask2Anomaly）
- 已知障碍物尺寸投影距离估计
- 未知区域底部接地点距离估计
- 基于道路掩码几何形状的自车行驶走廊预测
- 障碍物与走廊的空间关系、跨帧距离趋势和 TTC 风险计算
- 视频画面右侧显示障碍物距离和总风险的独立 UI
- 单个 JSON 文档、运行日志和结果视频输出
- 离线单元测试

## 4. 运行限制

- 完整 Mask2Anomaly 流程需要 Linux、NVIDIA GPU、Detectron2 和项目内的 CUDA 自定义算子。
- 模型权重、输入视频和输出文件均被 `.gitignore` 排除，需要在仓库的 releases 自行下载。
- 当前距离估计是单目几何估计，不是真实深度传感器测距。
- 当前风险评估只使用视频时间轴，没有车速、制动状态或 CAN 数据。

## 5. 目录结构

```text
vehicle-road-risk-warning/
├── checkpoints/              # 本地模型权重
├── configs/                  # 配置
├── data/raw/                 # 输入视频
├── docs/                     # 模块报告、接口说明和需求文档
├── mask2former/              # 内置 Mask2Anomaly/Mask2Former 模型代码
├── outputs/
│   ├── logs/                 # 运行日志
│   ├── results/              # 单个 JSON 结果文档
│   └── visualization/        # 结果视频
├── scripts/                  # 数据、评估、权重下载与 FP16 导出工具
├── src/
│   ├── config/               # 道路分割配置结构
│   ├── data/                 # 视频读取
│   ├── interface/            # 模块间数据结构和接口
│   ├── modules/              # 所有模块
│   ├── utils/                # 可视化工具
│   └── main.py               # 主流程
└── tests/                    # 离线单元测试
```

## 6. Linux 环境配置

### 6.1 已验证的环境组合

| 组件 | 版本 |
|---|---|
| Python | 3.10 |
| Conda CUDA Toolkit | 11.8 |
| PyTorch | 2.0.1+cu118 |
| Torchvision | 0.15.2+cu118 |
| NumPy | 1.26.4 |
| OpenCV | 4.10.0.84 |
| Transformers | 4.36.2 |
| Tokenizers | 0.15.2 |
| Hugging Face Hub | 0.20.3 |
| Detectron2 | 0.6，本地源码安装 |

不要在环境配置完成后随意升级 PyTorch、Torchvision、NumPy、Transformers 或 Detectron2。更换 PyTorch 后必须重新安装 Detectron2 并重新编译 CUDA 扩展。

### 6.2 检查系统和 GPU

```bash
uname -a
nvidia-smi
nvcc --version
```

Ubuntu 安装基础构建工具：

```bash
sudo apt update
sudo apt install -y build-essential cmake git ninja-build
```

### 6.3 创建 Conda 环境

```bash
conda create -n vrrw python=3.10 -y
conda activate vrrw

which python
python --version
python -m pip --version
```

安装构建工具，并保留包含 `pkg_resources` 的 Setuptools 版本：

```bash
python -m pip install --upgrade pip
python -m pip install \
  "setuptools==69.5.1" \
  "wheel>=0.40" \
  "ninja>=1.11"
```

### 6.4 安装并启用 Conda CUDA Toolkit 11.8

```bash
conda install -y \
  -c nvidia/label/cuda-11.8.0 \
  cuda-toolkit

export CUDA_HOME="$CONDA_PREFIX"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$CONDA_PREFIX/lib64:$CONDA_PREFIX/lib/python3.10/site-packages/torch/lib:${LD_LIBRARY_PATH:-}"

which nvcc
nvcc --version
```

此时 `which nvcc` 应指向 `vrrw` Conda 环境，版本应为 11.8。为了每次激活环境时自动设置变量，可以执行：

```bash
mkdir -p "$CONDA_PREFIX/etc/conda/activate.d"

cat > "$CONDA_PREFIX/etc/conda/activate.d/vrrw_cuda.sh" <<'EOF'
export CUDA_HOME="$CONDA_PREFIX"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$CONDA_PREFIX/lib64:$CONDA_PREFIX/lib/python3.10/site-packages/torch/lib:${LD_LIBRARY_PATH:-}"
EOF

conda deactivate
conda activate vrrw
```

### 6.5 安装 PyTorch 和项目依赖

```bash
python -m pip install --no-cache-dir \
  "torch==2.0.1+cu118" \
  "torchvision==0.15.2+cu118" \
  --index-url https://download.pytorch.org/whl/cu118
```

先固定已验证的冲突敏感版本：

```bash
python -m pip install --no-cache-dir \
  "numpy==1.26.4" \
  "opencv-python==4.10.0.84" \
  "transformers==4.36.2" \
  "tokenizers==0.15.2" \
  "huggingface-hub==0.20.3"
```

再进入项目目录安装其余依赖：

```bash
cd ~/vrrw
python -m pip install -r requirements.txt
```

### 6.6 安装 Detectron2

```bash
python -m pip install "git+https://github.com/facebookresearch/detectron2.git@v0.6"
```

### 6.7 编译 MultiScaleDeformableAttention

```bash
cd ~/vrrw/mask2former/modeling/pixel_decoder/ops
MAX_JOBS=4 FORCE_CUDA=1 sh make.sh
cd ~/vrrw
```

### 6.8 完整环境检查

```bash
python -c 'import torch; print("PyTorch:", torch.__version__); print("PyTorch CUDA:", torch.version.cuda); print("GPU available:", torch.cuda.is_available()); print("GPU:", torch.cuda.get_device_name(0))'
python -c 'import cv2, numpy, torchvision; print("NumPy:", numpy.__version__); print("OpenCV:", cv2.__version__)'
python -c 'import detectron2; print("Detectron2 OK")'
python -c 'import torch; import MultiScaleDeformableAttention; print("MSDeformAttn OK")'
```

## 7. 模型文件

```text
checkpoints/
├── yolo_best.pt
├── segformer_mit-b2_cityscapes.pth
└── mask2anomaly/
    └── best_contrastive_fp16.pth
```

## 8. 运行项目

### 8.1 准备输入视频

默认输入路径：

```text
data/raw/demo.mp4
```

也可以在 [configs/default.yaml](configs/default.yaml) 中修改：

```yaml
input:
  source_type: "video"
  video_path: "data/raw/demo.mp4"
  frame_skip: 1
```

`source_type` 目前只允许 `video`。

### 8.2 运行程序

```bash
python -m src.main
```

### 8.3 输出文件

```text
outputs/visualization/demo_result.mp4
outputs/results/demo_result.json
outputs/logs/run.log
```

## 9. 配置说明

主要配置文件：

- [configs/default.yaml](configs/default.yaml)：输入输出、YOLO、距离、走廊、风险、Mask2Anomaly 和窗口配置。
- [configs/road_segmenter.yaml](configs/road_segmenter.yaml)：SegFormer 模型、标签、后处理、设备和质量阈值。
- [configs/mask2anomaly/anomaly_inference.yaml](Detectron2/Mask2Anomaly/anomaly_inference.yaml)：模型结构配置。

阈值应优先在配置文件中修改，重要参数包括：

| 配置段 | 主要参数 |
|---|---|
| `known_detector` | 模型路径、置信度、IoU、类别和设备 |
| `unknown_detector` | 权重、像素阈值、区域面积、道路 ROI、已知框排除 |
| `distance_estimator` | 焦距、相机高度、地平线和类别真实高度 |
| `corridor_predictor` | 道路宽度、走廊宽度比例和时序平滑 |
| `risk_evaluator` | 空间交叠、距离、TTC、跟踪和连续帧确认阈值 |

距离参数与安装位置有关。更换视频分辨率、摄像机或安装高度后，应重新标定`focal_length_px`、`camera_height_m` 和 `horizon_ratio`。

## 10. 模块说明文档

- [未知障碍物检测模块](docs/UNKNOWN_DETECTOR_REPORT.md)
- [自车行驶走廊预测模块](docs/CORRIDOR_PREDICTOR_REPORT.md)
- [风险评估模块](docs/RISK_EVALUATOR_REPORT.md)
- [距离与走廊接口说明](docs/MODULE_INTERFACE_GUIDE.md)
- [更新日志](docs/CHANGELOG.md)
- [代码与 Git 规范](CONTRIBUTING.md)