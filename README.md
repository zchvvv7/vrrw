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
| Conda CUDA Toolkit | 11.8 |
| PyTorch | 2.0.1+cu118 |
| Torchvision | 0.15.2+cu118 |
| NumPy | 1.26.4 |
| OpenCV | 4.10.0.84 |
| Transformers | 4.36.2 |
| Tokenizers | 0.15.2 |
| Hugging Face Hub | 0.20.3 |
| Detectron2 | 0.6，本地源码安装 |


#### 2. 检查Linux和GPU

```bash
uname -a
nvidia-smi
nvcc --version
```

编译CUDA扩展需要C/C++编译器、CMake和Ninja。Ubuntu系统可以执行：

```bash
sudo apt update
sudo apt install -y build-essential cmake git ninja-build
```


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


#### 7. 编译MultiScaleDeformableAttention

项目真实构建脚本位于：

```text
mask2former/modeling/pixel_decoder/ops/make.sh
```

开始编译：

```bash
cd mask2former/modeling/pixel_decoder/ops
MAX_JOBS=4 FORCE_CUDA=1 sh make.sh
cd ~/vrrw
```


#### 8. 放置模型权重

项目使用三套模型，文件扩展名并不相同：

| 模块 | 参数文件                                            |
|---|-------------------------------------------------|
| YOLO已知障碍物检测 | `checkpoints/yolo_best.pt`                      |
| Mask2Anomaly未知障碍物检测 | `checkpoints/mask2anomaly/best_contrastive.pth` |
| SegFormer道路分割 | `checkpoints/segformer_mit-b2_cityscapes.pth`   |

#### 9. 运行完整流程

将测试视频放到：

```text
data/raw/demo.mp4
```

确认`configs/default.yaml`中的输入路径正确，然后执行：

```bash
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