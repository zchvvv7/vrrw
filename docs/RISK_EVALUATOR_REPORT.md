# 障碍物冲突与风险评估模块

## 实验报告

### 1.1 模型与方法

#### 1.1.1 模块目标

风险评估模块负责把检测结果转换为驾驶风险等级，它可以体现：

1. 障碍物是否占用或接近自车未来行驶走廊。
2. 障碍物与自车的距离是否已经过近。
3. 连续视频帧中的距离变化是否表明碰撞时间正在缩短。

当前实现是基于规则和连续帧状态的评估器，输入来自已知检测、未知检测、距离估计和自车走廊预测模块。

#### 1.1.2 障碍物占用区域

风险计算不能直接用整个检测框判断碰撞，因为框的上半部分通常是目标外观而不是接地点。

对于已知目标，模块只取检测框底部 footprint_height_ratio 比例作为近似路面占用区域。默认值为 0.25，即使用框底部 25%。

对于未知区域，模块优先解码列优先 RLE 掩码，直接使用真实异常区域；如果 RLE 缺失或无效，才退化为检测框底部占用区域。

已知目标进入风险计算前会经过合理性过滤：

| 过滤项 | 默认条件 | 目的 |
|---|---:|---|
| 检测置信度 | 不低于 0.45 | 去除低置信度误检 |
| 框面积占比 | 不高于 0.40 | 去除覆盖画面过大的异常框 |
| 接触画面边界数 | 不多于 2 | 去除明显截断或异常框 |

#### 1.1.3 空间冲突判断

模块把障碍物占用区域与自车走廊掩码进行比较。重叠率定义为：

~~~text
corridor_overlap = 障碍物占用区域与走廊的交集像素数 / 障碍物占用区域像素数
~~~

空间关系分为三类：

| 关系 | 判定 |
|---|---|
| intersecting | 重叠率达到阈值，并且障碍物底部接地点位于走廊中央有效区 |
| near | 未满足 intersecting，但与走廊膨胀后的缓冲区相交 |
| outside | 与走廊及其缓冲区均无有效冲突 |

默认重叠阈值 intersection_ratio 为 0.15。

仅靠矩形框重叠容易把路边目标误判为危险，因此模块还估计障碍物接地点：

1. 取占用区域最底部约 5% 的横坐标。
2. 用这些横坐标的中位数作为 contact_x。
3. 检查 contact_x 在障碍物最底行是否落入走廊。
4. 计算它相对该行走廊中心的横向偏移。
5. 偏移比例不超过 max_corridor_lateral_ratio 才认为真正进入走廊。

默认有效横向比例为 0.65。这个约束用于减少画面边缘框、车道旁目标和走廊边缘轻微相交造成的误报。

#### 1.1.4 走廊附近区域

near 关系通过对走廊掩码进行膨胀得到。膨胀半径按图像宽度计算：

~~~text
margin_px = round(frame_width * near_margin_ratio)
~~~

默认 near_margin_ratio 为 0.025。它让系统能够对尚未进入走廊、但已经靠近行驶路径的障碍物给出较低等级提醒。

#### 1.1.5 轻量级跨帧关联

为了避免单帧误检直接触发高风险，模块会把当前目标与历史目标关联。关联规则为：

- 来源必须相同，即 known 或 unknown。
- 已知目标的类别必须相同。
- 当前框与历史框的 IoU 不低于 track_iou_threshold。
- 同一历史轨迹在一帧内只能匹配一次。

匹配成功后累计 stable_frames；超过 max_track_gap_s 没有再次出现的轨迹会被删除。默认目标至少稳定出现 3 帧后，才允许进入 warning 或 danger。新出现或不稳定目标只给 notice。

轨迹 ID 形如 known-track-0001 或 unknown-track-0001。它只在本次视频运行期间有效，开始处理新视频前必须调用 reset。

#### 1.1.6 TTC 估计

TTC 是 Time To Collision，即按当前接近趋势继续运动时的预计碰撞时间。模块使用距离历史做一元线性拟合：

~~~text
distance(t) = slope * t + intercept
closing_speed = -slope
TTC = current_distance / closing_speed
~~~

只有同时满足以下条件，TTC 才被认为有效：

| 条件 | 默认值 |
|---|---:|
| 有效距离样本数 | 至少 8 |
| 观测时长 | 至少 0.25 秒 |
| 距离下降样本比例 | 至少 0.75 |
| 线性拟合 R² | 至少 0.80 |
| 接近速度 | 0.5 到 25.0 m/s |

这些门槛用于过滤单目距离抖动。若距离没有稳定下降、拟合质量不够或接近速度不合理，模块返回 TTC=None，不使用该 TTC 升级风险。

#### 1.1.7 风险分级

风险等级从低到高为 safe、notice、warning、danger。

首先应用通用规则：

| 条件 | 结果 |
|---|---|
| 障碍物在走廊外 | safe |
| 目标稳定帧数不足 | notice |

对 intersecting 障碍物，按顺序判断：

| 条件 | 结果 |
|---|---|
| 距离不大于 6 m | danger |
| 距离不大于 15 m 且 TTC 不大于 1.5 s | danger |
| 距离不大于 15 m | warning |
| 距离不大于 30 m 且 TTC 不大于 3 s | warning |
| 距离不可用 | warning |
| 其余情况 | notice |

对 near 障碍物：

| 条件 | 结果 |
|---|---|
| 距离不大于 6 m，或满足 warning TTC 条件 | warning |
| 其余情况 | notice |

当前帧最终风险取所有障碍物中的最高等级。如果没有障碍物，结果为 safe，原因是 no_obstacle；如果所有障碍物都在走廊外，结果也为 safe。

### 1.2 软件架构

#### 1.2.1 类与职责

~~~text
RiskEvaluator
├── 校验输入与走廊可用性
├── 过滤不可信的已知目标
├── 构造已知/未知障碍物占用掩码
├── 计算 intersecting / near / outside
├── 将当前目标关联到历史轨迹
├── 根据距离历史估计 TTC
├── 判定每个障碍物风险
└── 汇总当前帧最高风险
~~~

#### 1.2.2 单帧数据流

~~~text
known_objects ─┐
               ├──> 统一风险候选 ──> 空间冲突
unknown_regions┘                         │
                                         ├──> 轨迹关联与 TTC
corridor_result ─────────────────────────┘
                                                │
system_status ──────────────────────────────────┤
                                                ▼
                                     RiskEvaluationResult
                                                │
                                      可视化 + JSON 结果导出
~~~

#### 1.2.3 输入

RiskEvaluator.evaluate 接收：

| 输入 | 说明 |
|---|---|
| frame_id | 当前视频帧编号 |
| fps | 视频帧率，用于换算轨迹时间 |
| corridor_result | 自车走廊掩码、置信度和状态 |
| known_objects | 已知障碍物及其距离 |
| unknown_regions | 未知障碍物区域及其距离 |
| system_status | 当前感知系统状态 |

风险评估完全基于视频及其上游结果，不使用 Camera 对象，也不依赖实时摄像头输入。

#### 1.2.4 输出

RiskEvaluationResult 主要字段：

| 字段 | 说明 |
|---|---|
| risk_level | 当前帧最高风险等级 |
| major_reason | 触发该等级的主要规则 |
| obstacle_risks | 每个障碍物的独立风险结果 |
| system_status | normal、degraded 或 unavailable |
| is_valid | 本帧风险结果是否有效 |
| inference_time_ms | 规则计算耗时 |
| error_code / error_message | 错误信息 |
| model_version | 风险规则版本 |

ObstacleRisk 包含轨迹 ID、来源、类别、框、距离、走廊重叠率、空间关系、TTC、风险等级、原因和稳定帧数。

### 1.3 当前配置

默认配置位于 configs/default.yaml：

~~~yaml
risk_evaluator:
  enabled: true
  model_version: "video-risk-v1.0"
  intersection_ratio: 0.15
  near_margin_ratio: 0.025
  footprint_height_ratio: 0.25
  max_corridor_lateral_ratio: 0.65
  min_known_confidence: 0.45
  max_known_bbox_area_ratio: 0.40
  max_known_bbox_border_count: 2
  notice_distance_m: 30.0
  warning_distance_m: 15.0
  danger_distance_m: 6.0
  notice_ttc_s: 5.0
  warning_ttc_s: 3.0
  danger_ttc_s: 1.5
  track_iou_threshold: 0.30
  history_size: 15
  confirm_frames: 3
  min_ttc_samples: 8
  min_ttc_observation_s: 0.25
  min_ttc_r_squared: 0.80
  min_closing_observations_ratio: 0.75
  min_corridor_confidence: 0.45
  min_closing_speed_mps: 0.5
  max_closing_speed_mps: 25.0
  max_track_gap_s: 0.5
~~~

| 参数组 | 主要作用 |
|---|---|
| intersection / near | 控制障碍物与走廊的空间关系 |
| footprint / lateral | 控制接地点与有效占用区域 |
| min_known / max_known | 过滤已知检测误报 |
| distance thresholds | 控制不同距离对应的风险等级 |
| TTC thresholds | 控制快速接近目标的风险升级 |
| track / history | 控制跨帧关联和稳定性 |
| min_ttc_* | 控制 TTC 是否可信 |
| min_corridor_confidence | 拒绝低置信度走廊 |

notice_ttc_s 当前会被读取并参与配置大小关系校验，但尚未在 _classify_obstacle 的分级规则中实际使用。调整这个参数不会改变现有风险输出；如果未来增加 TTC notice 规则，应同时补充测试和本文档。

### 1.4 运行结果与可视化

#### 1.4.1 画面表现

可视化会叠加自车走廊，并按风险等级显示障碍物和帧级状态。当前颜色语义为：

| 等级 | 常用颜色 | 含义 |
|---|---|---|
| safe | 绿色 | 不在当前行驶冲突范围 |
| notice | 黄色 | 需要关注，但证据不足以报警 |
| warning | 橙色 | 已接近或进入走廊 |
| danger | 红色 | 距离或 TTC 达到紧急条件 |

障碍物级结果和帧级结果要区分：帧级风险是所有 obstacle_risks 的最大值。

#### 1.4.2 风险等级与系统状态

risk_level 表示道路画面中的碰撞风险；system_status 表示感知链路是否可靠。两者不是同一个概念。

| 情况 | risk_level | system_status / is_valid |
|---|---|---|
| 无障碍物且感知正常 | safe | normal / true |
| 有可靠危险目标 | danger | normal 或 degraded / true |
| 风险模块关闭 | safe | 继承系统状态 / false |
| 走廊不可用 | notice | degraded / false |
| 上游系统不可用 | notice | unavailable / false |
| 输入类型错误 | notice | unavailable / false |

因此，走廊不可用时返回 notice 是保守的“结果不可判定”，不能解释为画面里已经存在危险障碍物。使用端应同时查看 is_valid、system_status 和 major_reason。

#### 1.4.3 错误码

| 错误码 | 含义 |
|---:|---|
| 0 | 成功，或以明确状态返回不可用结果 |
| -1 | 输入无效 |
| -3 | 风险计算过程中发生异常 |

#### 1.4.4 测试范围

项目包含针对风险分级、走廊空间关系、接地点约束、已知框过滤、跨帧确认、TTC 拟合、不可用状态和主流程输出的测试。测试用于保证规则实现不被修改破坏，不代表阈值已经适配所有道路、天气和视频视角。

### 1.5 优势与局限

#### 优势

- 规则透明，每个风险结果都带 major_reason。
- 同时支持已知和未知障碍物。
- 使用占用区域与接地点，而不是只看矩形框重叠。
- 通过连续帧确认抑制单帧误报。
- TTC 只有在趋势质量满足条件时才启用。
- 风险状态与感知系统状态分离，便于降级处理。

#### 局限

- 单目距离误差会直接影响距离阈值和 TTC。
- IoU 关联不是完整多目标跟踪器，遮挡和快速运动时可能换 ID。
- 固定阈值未考虑车速、道路限速、制动能力和天气。
- 自车走廊误差会传递到空间冲突判断。
- 仅根据视频帧编号和 FPS 建立时间轴，没有车辆 CAN 数据。
- near 目标的极近距离只升到 warning，属于当前保守规则设计。
- notice_ttc_s 当前没有参与实际分级。

### 1.6 使用与排查

#### 1.6.1 正确调用顺序

每个视频开始前：

~~~python
risk_evaluator.reset()
~~~

每帧应在已知检测、未知检测、距离估计和走廊预测完成后调用 evaluate。known_objects 和 unknown_regions 中的 distance 应尽可能在调用前填充。

#### 1.6.2 阅读结果

排查风险结果时建议按以下顺序检查：

1. is_valid 是否为 true。
2. system_status 是否为 normal。
3. major_reason 指向哪条规则。
4. 最高风险 obstacle 的 spatial_relation。
5. corridor_overlap、distance、ttc 和 stable_frames。
6. 画面中的走廊掩码是否合理。

只看 risk_level 容易把“系统不可用的保守 notice”误认为真实道路风险。

#### 1.6.3 常见现象

| 现象 | 优先检查 |
|---|---|
| 空画面出现 danger | 是否存在误检框、距离异常、旧轨迹是否 reset |
| 路边目标被判 intersecting | 走廊掩码、接地点、lateral ratio |
| 目标刚出现只显示 notice | confirm_frames，属于预期抖动抑制 |
| TTC 一直为 None | 样本数、距离趋势、R²、接近速度范围 |
| 走廊不可用却显示 notice | 查看 is_valid=false 与 major_reason |
| unknown 风险没有距离 | 未知区域距离估计是否在风险调用之前 |
| 风险不断跳变 | 目标关联、距离抖动、走廊稳定性 |

### 1.7 后续优化建议

1. 用带真实碰撞风险标注的视频校准距离与 TTC 阈值。
2. 引入更可靠的多目标跟踪器，处理遮挡和 ID 切换。
3. 对距离序列使用鲁棒滤波或不确定性区间。
4. 将自车速度、制动距离和道路曲率加入动态阈值。
5. 为走廊置信度和距离置信度建立联合降级策略。
6. 明确定义并实现 notice_ttc_s 的业务规则，或删除无效配置。
7. 分开评估误报、漏报、风险提前量和等级稳定性。

---

## 2. 系统集成说明

### 2.1 主流程位置

风险模块位于逐帧感知链路末端：

~~~text
视频读取
  -> 道路分割
  -> 已知/未知障碍物检测
  -> 距离估计
  -> 自车走廊预测
  -> 风险评估
  -> 可视化与结果导出
~~~

### 2.2 上下游接口

| 方向 | 模块 | 关键数据 |
|---|---|---|
| 上游 | KnownDetector | 类别、框、置信度、距离 |
| 上游 | UnknownDetector | 区域、RLE、框、距离 |
| 上游 | CorridorPredictor | corridor_mask、confidence、状态 |
| 下游 | Visualizer | 帧级与障碍物级风险 |
| 下游 | JSON 导出 | 风险原因、TTC、空间关系、系统状态 |

### 2.3 修改约束

- 修改风险等级字符串时，应同步 schemas、可视化和导出端。
- 修改 RLE 规则时，应同步未知检测器和风险解码器。
- 修改距离单位时，必须同步所有距离与速度阈值；当前单位为米和米每秒。
- 修改跟踪规则后，应验证确认帧数与 TTC 历史是否仍然连续。
- 不应把系统不可用静默输出为 safe。
- 新增规则必须给出可解释 major_reason 并添加单元测试。

---

## 附录：关键代码位置

| 内容 | 文件 |
|---|---|
| 风险评估核心 | src/modules/risk_evaluator.py |
| 风险与障碍物数据结构 | src/interface/schemas.py |
| 风险参数 | configs/default.yaml |
| 主流程接入 | src/main.py |
| 自车走廊预测 | src/modules/corridor_predictor.py |
| 距离估计 | src/modules/distance_estimator.py |
| 已知障碍物检测 | src/modules/known_detector.py |
| 未知障碍物检测 | src/modules/unknown_detector.py |
| 结果绘制 | src/utils/result_visualizer.py |
| 实时窗口 | src/utils/live_visualizer.py |
| 风险相关测试 | tests/ |
