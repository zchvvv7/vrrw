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
- 未知障碍物检测（基于road mask找异常区域）
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

可通过以下指令安装：
pip install -r requirements.txt
```


### 配置修改

请直接在default.yaml中修改阈值，不要在代码中修改。


### 运行方法

- 将测试视频放入：data/raw/，默认输入路径为data/raw/demo.mp4
- 终端输入`python -m src.main`
- 在outputs/visualization/demo_result.mp4查看输出结果


### 当前版本

- 简单切割道路可行驶区域（未进行区域边缘平滑优化）
- 已知障碍物检测尚未接入
- 未知障碍物检测基于road mask，为初版逻辑