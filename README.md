## 车载视觉道路障碍物与未知风险预警系统


### 项目简介

本项目是一个车载视觉道路风险预警系统，用前视视频识别可行驶区域、已知障碍物和未知异常区域。系统会结合障碍物位置、距离和行驶风险，
输出 safe / notice / warning / danger 等风险等级，并生成可视化结果。


### 当前版本

- 系统版本：`vrrws_v1.0`
- 配置版本：`threshold_v1.0`


### 当前已实现功能

- 输入
  - 本地视频文件输入
- 道路可行驶区域分割
- 已知障碍物检测（YOLO）
- 未知障碍物检测（Mask2Anomaly像素级异常分割）
- 可视化结果输出


### 目录结构说明

```commandline
checkpoints/
    存放模型权重。
    
configs/
    存放系统配置文件，用于调整阈值参数。
    
data/
    data/raw/ 存放原始输入视频。
    data/processed/ [暂未使用]，存放处理后的数据。

docs/
    存放更新日志、项目说明等文件。

outputs/
    outputs/logs/ [暂未使用]，存放运行日志。
    outputs/results/ [暂未使用]，存放JSONL结果文件。
    outputs/visualization/ 存放可视化结果视频文件。

scripts/
    存放项目辅助命令脚本。

src/
    src/config/ 存放项目配置读取和配置数据结构。
    src/interface/ 存放模块之间统一传递的数据结构和接口约定。
    src/modules/ 存放项目主模块代码。
    src/utils/ 存放通用工具函数。

tests/
    存放测试代码。
```


### Linux环境配置与部署

本节为本项目在Linux/NVIDIA服务器上的完整部署流程。Mask2Anomaly依赖
Detectron2和自定义CUDA扩展，对PyTorch、CUDA、NumPy和Transformers版本较为
敏感，请严格按照本节顺序执行。

不要在安装完成后随意执行`pip install -U torch`、`pip install -U numpy`或
`pip install -U transformers`。如果更换PyTorch版本，Detectron2和
MultiScaleDeformableAttention都必须重新编译。

Apple Silicon不支持项目当前使用的CUDA扩展，只适合运行不依赖该扩展的单元测试，不能作为最终性能验收环境。


#### 1. 推荐环境版本

以下组合已经在Linux服务器上完成实际验证：

| 组件 | 版本 |
|---|---|
| Python | 3.10 |
| 系统CUDA | 12.0，可以保留 |
| Conda CUDA Toolkit | 11.8 |
| PyTorch | 2.0.1+cu118 |
| Torchvision | 0.15.2+cu118 |
| NumPy | 1.26.4 |
| OpenCV | 4.10.0.84 |
| Transformers | 4.36.2 |
| Tokenizers | 0.15.2 |
| Hugging Face Hub | 0.20.3 |
| Detectron2 | 0.6，本地源码安装 |

系统显示CUDA 12.0并不代表必须安装cu120版PyTorch。NVIDIA驱动通常可以向下
兼容CUDA 11.8运行时。本项目使用PyTorch cu118，并使用Conda环境内的CUDA
Toolkit 11.8编译自定义扩展。


#### 2. 检查Linux和GPU

```bash
uname -a
nvidia-smi
nvcc --version
```

如果系统`nvcc`显示CUDA 12.0，可以继续。后续会在Conda环境中安装11.8版本，
不会删除系统CUDA。

编译CUDA扩展需要C/C++编译器、CMake和Ninja。Ubuntu系统可以执行：

```bash
sudo apt update
sudo apt install -y build-essential cmake git ninja-build
```

没有`sudo`权限时，需要联系服务器管理员提供这些工具，或者加载服务器已有
的编译工具模块。


#### 3. 创建独立Conda环境

```bash
conda create -n vrrw python=3.10 -y
conda activate vrrw
```

确认当前Python确实来自新环境：

```bash
which python
python --version
python -m pip --version
```

路径应位于当前Conda环境，例如：

```text
$CONDA_PREFIX/bin/python
```

安装构建工具，并固定包含`pkg_resources`的Setuptools版本：

```bash
python -m pip install --upgrade pip
python -m pip install \
  "setuptools==69.5.1" \
  "wheel>=0.40" \
  "ninja>=1.11"
```

验证：

```bash
python -c 'import pkg_resources; print("pkg_resources: OK")'
```


#### 4. 安装Conda CUDA Toolkit 11.8

```bash
conda install -y \
  -c nvidia/label/cuda-11.8.0 \
  cuda-toolkit
```

让当前终端优先使用Conda环境内的CUDA：

```bash
export CUDA_HOME="$CONDA_PREFIX"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$CONDA_PREFIX/lib64:$CONDA_PREFIX/lib/python3.10/site-packages/torch/lib:${LD_LIBRARY_PATH:-}"
```

此时再次检查：

```bash
which nvcc
nvcc --version
```

`which nvcc`应指向`vrrw`环境，版本应为11.8。

每次重新进入终端都需要设置上述变量。也可以创建Conda激活脚本：

```bash
mkdir -p "$CONDA_PREFIX/etc/conda/activate.d"
```

```bash
cat > "$CONDA_PREFIX/etc/conda/activate.d/vrrw_cuda.sh" <<'EOF'
export CUDA_HOME="$CONDA_PREFIX"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$CONDA_PREFIX/lib64:$CONDA_PREFIX/lib/python3.10/site-packages/torch/lib:${LD_LIBRARY_PATH:-}"
EOF
```

重新激活环境使脚本生效：

```bash
conda deactivate
conda activate vrrw
```


#### 5. 安装PyTorch cu118

```bash
python -m pip install --no-cache-dir \
  "torch==2.0.1+cu118" \
  "torchvision==0.15.2+cu118" \
  --index-url https://download.pytorch.org/whl/cu118
```

验证PyTorch、CUDA运行时和GPU：

```bash
python - <<'PY'
import torch
from torch.utils.cpp_extension import CUDA_HOME

print("PyTorch:", torch.__version__)
print("PyTorch CUDA:", torch.version.cuda)
print("CUDA_HOME:", CUDA_HOME)
print("GPU available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
PY
```

预期关键结果：

```text
PyTorch: 2.0.1+cu118
PyTorch CUDA: 11.8
GPU available: True
```

这里`PyTorch CUDA`显示11.8，而系统`nvcc --version`最初显示12.0，是正常
现象。


#### 6. 安装Python依赖并锁定冲突版本

先安装本项目验证过的关键版本：

```bash
python -m pip install --no-cache-dir \
  "numpy==1.26.4" \
  "opencv-python==4.10.0.84" \
  "transformers==4.36.2" \
  "tokenizers==0.15.2" \
  "huggingface-hub==0.20.3"
```

再安装其余项目依赖：

```bash
cd ~/vrrw
python -m pip install -r requirements.txt
```

不要为该项目安装NumPy 2.x。PyTorch 2.0.1与NumPy 2.x组合可能出现：

```text
RuntimeError: Could not infer dtype of numpy.float32
```

不要安装过新的Transformers。新版本可能将PyTorch 2.0.1判断为不受支持，
并显示具有误导性的错误：

```text
SegformerImageProcessor requires the PyTorch library but it was not found
```

安装完成后确认版本：

```bash
python - <<'PY'
import cv2
import numpy
import torch
import torchvision
import transformers

print("NumPy:", numpy.__version__)
print("OpenCV:", cv2.__version__)
print("PyTorch:", torch.__version__)
print("Torchvision:", torchvision.__version__)
print("Transformers:", transformers.__version__)
print(torch.as_tensor(numpy.float32(1.0)))
PY
```

最后一行应正常输出`tensor(1.)`。


#### 7. 从本地源码安装Detectron2

服务器无法连接GitHub时，需要提前在其他电脑下载Detectron2源码并上传。
本教程假定目录结构如下：

```text
用户主目录/
├── detectron2/
└── vrrw/
```

安装本地Detectron2：

```bash
cd ~/vrrw
FORCE_CUDA=1 python -m pip install -e ~/detectron2
```

验证：

```bash
python - <<'PY'
import torch
import detectron2

print("PyTorch:", torch.__version__)
print("Detectron2:", detectron2.__version__)
print("Detectron2 import: OK")
PY
```

Detectron2必须在最终PyTorch版本确定后安装。更换PyTorch后，原先编译的
Detectron2不可继续使用。


#### 8. 编译MultiScaleDeformableAttention

项目真实构建脚本位于：

```text
mask2former/modeling/pixel_decoder/ops/make.sh
```

编译前再次检查：

```bash
cd ~/vrrw
echo "$CUDA_HOME"
which nvcc
nvcc --version
```

开始编译：

```bash
cd mask2former/modeling/pixel_decoder/ops
MAX_JOBS=4 FORCE_CUDA=1 sh make.sh
cd ~/vrrw
```

`MAX_JOBS=4`用于降低编译期间的内存压力。服务器内存较小时可以改为
`MAX_JOBS=2`。

验证扩展。应先导入PyTorch，再导入CUDA扩展：

```bash
python -c 'import torch; import MultiScaleDeformableAttention; print("MSDeformAttn OK")'
```

如果出现：

```text
ImportError: libc10.so: cannot open shared object file
```

说明动态库搜索路径没有生效。重新执行：

```bash
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$CONDA_PREFIX/lib64:$CONDA_PREFIX/lib/python3.10/site-packages/torch/lib:${LD_LIBRARY_PATH:-}"
```

然后重新运行验证命令。


#### 9. 放置模型权重

项目使用三套模型，文件扩展名并不相同：

| 模块 | 参数文件 |
|---|---|
| YOLO已知障碍物检测 | `checkpoints/yolo_best.pt` |
| Mask2Anomaly未知障碍物检测 | `checkpoints/mask2anomaly/best_contrastive.pth` |
| SegFormer道路分割 | `pytorch_model.bin`和项目微调`.pth` |

#### 10. 运行完整流程

将测试视频放到：

```text
data/raw/demo.mp4
```

确认`configs/default.yaml`中的输入路径正确，然后执行：

```bash
TRANSFORMERS_OFFLINE=1 \
HF_HUB_OFFLINE=1 \
python -m src.main
```

输出文件：

```text
outputs/visualization/demo_result.mp4
outputs/results/demo_result.json
outputs/logs/run.log
```


### 配置修改

请直接在default.yaml中修改阈值，不要在代码中修改。


### 当前版本

- 简单切割道路可行驶区域（未进行区域边缘平滑优化）
- 已知障碍物检测已接入YOLO
- 未知障碍物检测已接入Mask2Anomaly
- 风险状态机和距离估计尚未完成