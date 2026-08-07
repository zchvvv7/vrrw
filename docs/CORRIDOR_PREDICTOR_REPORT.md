# 自车走廊预测模块

## 实验报告

### 1.1 使用的模型与方法

#### 1.1.1 道路分割模型

| 属性 | 说明 |
|------|------|
| 模型名称 | SegFormer-B2 |
| 预训练权重 | `nvidia/segformer-b2-finetuned-cityscapes-1024-1024` |
| 微调权重 | `checkpoints/segformer_mit-b2_cityscapes.pth` |
| 输入尺寸 | 1024 × 1024 |
| 输出类别 | 19 类（Cityscapes） |
| 推理速度 | ~650ms/帧（CPU） |

#### 1.1.2 几何投影方法

**原理**：基于单目视觉的几何投影公式

```
distance = focal_length × camera_height / vertical_offset
```

其中：
- `focal_length`：相机焦距（像素）
- `camera_height`：相机离地高度（米）
- `vertical_offset`：目标点到地平线的像素距离

#### 1.1.3 走廊构建方法

**方法**：基于道路掩码的几何拟合

1. 从道路分割掩码提取左右边界
2. 拟合消失点（Vanishing Point）
3. 构建梯形/三角形走廊
4. 生成走廊中心线

### 1.2 封装架构

#### 1.2.1 类结构

```
CorridorPredictor
├── __init__(config)
│   ├── 基础参数配置
│   ├── 几何投影参数配置
│   └── 缓存初始化
│
├── predict(frame, frame_id, road_mask, 
│          known_objects=None, unknown_regions=None)
│   ├── _predict_corridor()          # 核心走廊预测
│   ├── _apply_obstacle_avoidance()  # 障碍物避让
│   └── _compute_prediction_markers()# 距离/时间计算
│
├── get_obstacle_avoidance_result()  # 获取避让结果
└── get_prediction_info()            # 获取预测信息
```

#### 1.2.2 数据流

```
输入帧 → SegFormer → 道路掩码 → 走廊预测 → 障碍物避让 → 输出
                                        ↓
                                   几何投影计算
                                        ↓
                                   距离/时间标记
```

#### 1.2.3 接口设计

| 接口 | 输入 | 输出 | 说明 |
|------|------|------|------|
| `predict()` | 帧、道路掩码、障碍物列表 | `CorridorPredictionResult` | 主预测接口 |
| `get_obstacle_avoidance_result()` | 无 | `np.ndarray` | 获取避让后掩码 |
| `get_prediction_info()` | 无 | `List[dict]` | 获取标记点列表 |

**标记点格式**：
```python
{
    "x": 640,           # 像素 x 坐标
    "y": 500,           # 像素 y 坐标
    "time_s": 0.77,     # 到达时间（秒）
    "distance_m": 10.71 # 真实距离（米）
}
```

### 1.3 配置参数

```yaml
corridor_predictor:
  enabled: true
  method: "road_mask_geometry"
  
  # 几何投影参数
  focal_length_px: 1000.0      # 相机焦距
  camera_height_m: 1.50        # 相机离地高度
  horizon_ratio: 0.50          # 地平线位置比例
  ego_speed_mps: 13.9          # 自车速度（50km/h）
  min_distance_m: 0.3          # 最小有效距离
  max_distance_m: 80.0         # 最大有效距离
```

### 1.4 实验结果

#### 1.4.1 距离计算验证

| 像素位置 (y) | 距地平线像素 | 计算距离 | 实际时间 (50km/h) |
|-------------|------------|---------|------------------|
| 400 | 40px | 37.50m | 2.70s |
| 500 | 140px | 10.71m | 0.77s |
| 600 | 240px | 6.25m | 0.45s |
| 700 | 340px | 4.41m | 0.32s |

#### 1.4.2 障碍物避让验证

- ✅ 已知障碍物 bbox 区域被正确扣除
- ✅ 未知区域 bbox 被正确扣除
- ✅ 空障碍物列表不影响正常功能
- ✅ 避让后面积正确减少

#### 1.4.3 测试覆盖

| 测试类别 | 用例数 | 通过率 |
|---------|-------|-------|
| 基础功能测试 | 18 | 100% |
| 障碍物避让测试 | 7 | 100% |
| 几何投影测试 | 8 | 100% |
| 配置参数测试 | 3 | 100% |
| **总计** | **36** | **100%** |

### 1.5 优缺点分析

#### 1.5.1 优点

1. **计算高效**
   - 几何投影为解析解，无需训练
   - 障碍物避让为简单像素操作
   - 总体增加开销 < 5ms

2. **参数可配**
   - 所有几何参数可通过配置文件调整
   - 支持不同车型/相机配置

3. **接口兼容**
   - 新增参数均为可选参数
   - 向下兼容现有接口
   - 不修改 `schemas.py`

4. **数据分离**
   - 预测逻辑与可视化分离
   - 格式由调用方决定
   - 便于单元测试

5. **鲁棒性**
   - 地平线以上点自动过滤
   - 超距点自动过滤
   - 边界情况有合理默认值

#### 1.5.2 缺点

1. **距离估计精度有限**
   - 依赖单目几何，无深度信息
   - 假设平坦地面，坡道场景会有误差
   - 焦距/相机高度需准确标定

2. **障碍物避让简单**
   - 仅使用 bbox 矩形扣除
   - 不规则障碍物会过度扣除
   - 可考虑使用实例分割 mask

3. **速度假设**
   - 自车速度为配置值，非实时获取
   - 实际使用需对接 CAN 总线或 GPS
   - 动态场景下可能过时

4. **无时间平滑**
   - 逐帧独立计算，无时序平滑
   - 相邻帧预测可能有抖动
   - 可考虑卡尔曼滤波

### 1.6 使用指南

#### 1.6.1 基础使用（仅预测走廊）

```python
predictor = CorridorPredictor(config)
result = predictor.predict(frame, frame_id, road_mask)
# 返回 corridor_mask, polygon, centerline, confidence
```

#### 1.6.2 启用障碍物避让

```python
result = predictor.predict(
    frame=frame,
    frame_id=frame_id,
    road_mask=road_mask,
    known_objects=detected_objects,      # 可选
    unknown_regions=unknown_regions     # 可选
)
avoided_mask = predictor.get_obstacle_avoidance_result()
```

#### 1.6.3 获取预测信息

```python
predictor.predict(frame, frame_id, road_mask)
prediction_info = predictor.get_prediction_info()
# [{"x": 640, "y": 500, "time_s": 0.77, "distance_m": 10.71}, ...]
```

#### 1.6.4 自定义可视化

```python
info = predictor.get_prediction_info()
for marker in info:
    x, y = marker["x"], marker["y"]
    # 自由设置格式
    text = f"{marker['time_s']:.1f}s / {marker['distance_m']:.1f}m"
    # 绘制逻辑...
```

### 1.7 后续工作建议

1. **集成实时速度**：对接车辆 CAN 总线或 GPS，获取真实自车速度
2. **多传感器融合**：结合 LiDAR 或深度相机提升距离估计精度
3. **时序平滑**：对预测结果使用滑动窗口或卡尔曼滤波
4. **实例分割集成**：使用 YOLO-Seg 或 Mask2Former 获取精确障碍物 mask
5. **自适应参数**：根据场景动态调整焦距/地平线位置

---

## 二、关于梯形与三角形走廊的变化条件

在 `_build_trapezoid_corridor` 方法中，走廊形状的变化主要取决于**道路边缘拟合的消失点位置**和**配置参数**：

### 🔺 三角形情况
当满足以下任一条件时，走廊会退化为**三角形**（即两边在远处汇聚成一个点）：
1. **配置导致**：`corridor_top_ratio >= 1.0`
    - 这个参数控制走廊顶端在画面中的位置。当比例为 1.0 时，顶端刚好位于消失点 `vp_y`，此时顶边宽度为 0，形成三角形。
2. **拟合失败/回退逻辑**：
    - 当左右道路边缘拟合的消失点 `vp_y` 落在画面底部以下（`vp_y >= road_bottom_y`）时，会触发回退逻辑。
    - 回退逻辑会强制假设一个在画面上方的消失点，并以车头中心为基准构建一个对称形状。

### 🔻 梯形情况（默认）
在正常情况下（`corridor_top_ratio < 1.0` 且拟合正常），走廊表现为**梯形**：
- 底边较宽（对应车头位置的道路宽度）
- 顶边较窄（对应预测范围尽头的道路宽度）
- 这是基于近大远小的透视原理构建的

---

## 三、集成说明

### 3.1 配置文件设置

在 `configs/default.yaml` 的 `corridor_predictor` 部分添加几何投影参数：

```yaml
corridor_predictor:
  enabled: true
  # ... 其他参数 ...
  
  # 几何投影参数（新增）
  focal_length_px: 1000.0
  camera_height_m: 1.50
  horizon_ratio: 0.50
  ego_speed_mps: 13.9
  min_distance_m: 0.3
  max_distance_m: 80.0
```

---

## 附录：关键代码位置

| 文件 | 内容 |
|------|------|
| `src/modules/corridor_predictor.py` | 主实现文件 |
| `tests/test_corridor_predictor.py` | 测试文件 |
| `configs/default.yaml` | 配置文件 |
