# 距离估计与视频行驶走廊模块开发接口

本文档用于把两个模块分别交给不同开发人员。总流程、输入输出结构、
配置入口和占位调用已经接好，开发人员不需要修改 `src/main.py`。

## 共同约束

- 系统输入是视频文件逐帧读取的 BGR 图像。
- 不接入摄像头参数、CAN、车速、方向盘转角或横摆角速度。
- 所有距离单位统一为米。
- 所有图像坐标采用左上角为原点的像素坐标。
- 不得伪造距离、置信度或走廊结果。
- 模块失败时必须返回已有错误结构，不能使整条视频流程崩溃。
- Python 代码必须遵守项目根目录的 `CONTRIBUTING.md`。

## 任务一：已知障碍物距离估计

### 允许修改的文件

- `src/modules/distance_estimator.py`
- `tests/test_distance_estimator.py`
- `configs/default.yaml` 中的 `distance_estimator` 配置段
- 算法需要的新配置文件、依赖说明和测试资源

除非与项目负责人确认，不要修改 `src/main.py`、公共数据结构或未知
障碍物检测模块。

### 固定入口

```python
def estimate(
    self,
    frame: np.ndarray,
    frame_id: int,
    known_objects: List[DetectedObject],
) -> DistanceEstimationResult:
```

本阶段只处理 `KnownDetector` 输出的 `known_objects`。接口中没有
`UnknownRegion` 或 `unknown_regions` 参数，禁止在本模块内调用未知
障碍物检测器。未知障碍物距离以后由项目负责人统一接入。

### 固定输出

`DistanceEstimationResult.known_objects` 必须满足：

- 数量与输入 `known_objects` 完全一致；
- 顺序与输入完全一致；
- `class_name`、`bbox` 和 `confidence` 不得被修改；
- 成功估计时只更新 `DetectedObject.distance`，单位为米；
- 无法可靠估计时将该目标的 `distance` 保持为 `None`；
- 不得用 `0`、框高度或相对深度值冒充米制距离。

具体算法和模型由负责人选型，当前工程没有指定
`depth_anything_v2_vits.pth`，也不得假设该权重已经接入主流程。
如果使用学习模型，必须在配置中声明项目相对权重路径、设备和版本。

### 实现位置

在 `DistanceEstimator._estimate_known_objects()` 中实现算法。外层
`estimate()` 已经负责：

- 输入校验；
- 推理耗时统计；
- 异常转错误码；
- 保持失败时的原始检测结果；
- 向主流程回传统一结构。

### 最低验收要求

- 单个目标、多个目标、空目标列表均可运行；
- 输出数量和顺序严格不变；
- 边界框越界或面积过小时不会崩溃；
- 同一帧的深度模型最多推理一次，不能每个目标重复推理；
- 测试集同时包含近、中、远距离，并给出绝对误差和相对误差；
- `pytest tests/test_distance_estimator.py` 全部通过。

完成后将 `configs/default.yaml` 中：

```yaml
distance_estimator:
  enabled: true
  method: "实际算法名称"
  model_version: "实际模型或算法版本"
```

## 任务二：视频自车行驶走廊预测

### 允许修改的文件

- `src/modules/corridor_predictor.py`
- `tests/test_corridor_predictor.py`
- `configs/default.yaml` 中的 `corridor_predictor` 配置段
- 算法需要的新配置文件和测试资源

除非与项目负责人确认，不要修改 `src/main.py`、道路分割输出结构或
风险模块。

### 固定入口

```python
def predict(
    self,
    frame: np.ndarray,
    frame_id: int,
    road_mask: np.ndarray,
) -> CorridorPredictionResult:
```

该模块预测的是视频图像空间中的自车行驶走廊，不是基于车速和转角的
车辆运动轨迹。允许使用当前帧、道路掩码和模块内部保存的历史视频帧
状态。

每次开始处理新视频时，总流程会调用：

```python
corridor_predictor.reset()
```

实现若使用跨帧平滑，必须在 `reset()` 中清空全部历史状态。

### 固定输出

- `corridor_mask`：与输入帧同高同宽的二维 `uint8` 掩码；
- `polygon`：按顺序排列的走廊边界像素点；
- `centerline`：由近到远或由远到近排列的中心线像素点；
- `confidence`：范围为 `[0.0, 1.0]`；
- 无可靠结果时不得伪造走廊，应返回失败结果或空结果。

走廊必须限制在有效道路区域内，并处理道路分割缺失、断裂、窄区域和
弯道场景。建议先逐行提取道路左右边界，再拟合中心线和走廊宽度，
最后使用视频前后帧进行稳定处理。

### 实现位置

在 `CorridorPredictor._predict_corridor()` 中实现算法，并在需要时
实现 `reset()`。外层 `predict()` 已经负责输入校验、耗时统计、
输出尺寸校验和异常转错误码。

### 最低验收要求

- 输出掩码尺寸始终与视频帧一致；
- 输出置信度始终处于 `[0.0, 1.0]`；
- 走廊主体不超出道路掩码；
- 连续视频中不存在明显的逐帧左右跳动；
- 道路掩码无效时不会崩溃，也不会输出虚假高置信度走廊；
- `reset()` 后不保留上一段视频的历史状态；
- `pytest tests/test_corridor_predictor.py` 全部通过。

完成后将 `configs/default.yaml` 中：

```yaml
corridor_predictor:
  enabled: true
  method: "实际算法名称"
  model_version: "实际模型或算法版本"
```

## 已完成的总流程接线

当前每帧调用顺序如下：

```text
视频帧
  -> 道路分割
  -> 已知障碍物检测
  -> 已知障碍物距离估计
  -> 未知障碍物检测
  -> 视频行驶走廊预测
  -> 可视化与 JSON 输出
```

距离估计返回的 `known_objects` 会继续传给下游；未知障碍物仅继续参与
原有未知检测、显示和记录，不会进入距离估计模块。走廊结果会写入
`FrameResult` 和 JSON，未来可直接供空间冲突与风险计算使用。
