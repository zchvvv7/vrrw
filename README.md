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


### 环境要求

```commandline
numpy>=1.24.0
opencv-python>=4.8.0
torch>=2.0.0
torchvision>=0.15.0
transformers>=4.35.0
segmentation-models-pytorch>=0.3.3
Pillow>=10.0.0
pyyaml>=6.0.0
pytest>=7.0.0
fvcore>=0.1.5
iopath>=0.1.9
timm>=0.6.13
pycocotools>=2.0.7
tabulate>=0.9.0

可通过以下指令安装：
pip install -r requirements.txt
```

Mask2Anomaly还依赖Detectron2和MSDeformAttn扩展。需要根据目标服务器的
PyTorch、CUDA版本安装Detectron2，并在Linux/NVIDIA环境执行：

```commandline
cd mask2former/modeling/pixel_decoder/ops
sh make.sh
```

Apple Silicon不支持该项目当前使用的CUDA扩展，不能作为最终性能验收环境。


### Mask2Anomaly模型

- 模型配置：`configs/mask2anomaly/anomaly_inference.yaml`
- 模型权重：`checkpoints/mask2anomaly/best_contrastive.pth`
- 模型源码：`mask2former/`
- 模型后端：`src/modules/mask2anomaly_backend.py`
- 未知区域后处理：`src/modules/unknown_detector.py`

模型权重不提交Git，交付时应通过单独权重包或受控存储提供。


### 配置修改

请直接在default.yaml中修改阈值，不要在代码中修改。


### 运行方法

- 将测试视频放入：data/raw/，默认输入路径为data/raw/demo.mp4
- 终端输入`python -m src.main`
- 在outputs/visualization/demo_result.mp4查看输出结果


### 当前版本

- 简单切割道路可行驶区域（未进行区域边缘平滑优化）
- 已知障碍物检测尚未接入
- 未知障碍物检测已接入Mask2Anomaly
- 风险状态机和距离估计尚未完成