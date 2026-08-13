# 未知障碍物检测模块

## 实验报告

### 1.1 模型与方法

#### 1.1.1 模块目标

未知障碍物检测模块负责寻找已知目标检测器没有识别到、但可能占据道路的异常区域。它先判断每个像素是否偏离已知道路场景分布，再结合道路区域和已知目标结果生成未知障碍物候选框。

当前实现由以下部分组成：

| 部分 | 实现 | 作用 |
|---|---|---|
| 主模型 | Mask2Anomaly / Mask2Former | 生成语义掩码类别概率 |
| 推理框架 | Detectron2 | 加载配置、权重并执行推理 |
| 像素解码器 | MSDeformAttnPixelDecoder | 融合多尺度视觉特征 |
| Transformer 解码器 | MultiScaleMaskedTransformerDecoder_GMA | 输出类别概率与掩码 |
| 后处理 | 阈值、道路 ROI、形态学与轮廓提取 | 将异常分数图变成区域 |
| 已知目标抑制 | 已知框扩张后清零 | 避免重复报告 YOLO 已识别目标 |

模型配置位于 configs/mask2anomaly/anomaly_inference.yaml。当前骨干网络为 ResNet-50，语义类别数为 19，查询数为 100。模块只执行推理，不在主流程内训练。

#### 1.1.2 异常分数

Mask2Anomaly 后端取得模型的语义输出，并只把前 19 个 Cityscapes 已知类别视为正常类别。每个像素的异常分数定义为：

~~~text
anomaly_score(x, y) = 1 - max(inlier_scores(x, y))
~~~

分数越接近 1，表示该像素越难由任意已知类别解释；越接近 0，表示它更像已知道路场景。

如果模型额外返回未知或无效类别通道，后端会根据 known_mask_threshold 抑制已经被强已知响应覆盖的区域。最终分数被限制到 0 到 1，并转换为 CPU float32 数组，保证后续 OpenCV 处理稳定。

需要注意：异常高分不等同于真实障碍物。阴影、反光、道路纹理、图像模糊和域差异也可能产生高分，因此必须经过后处理。

#### 1.1.3 道路区域约束

模块不会在整幅图像中无条件寻找异常点。它利用道路分割模块提供的 road_mask，并执行以下约束：

1. 将道路掩码转换为二值图。
2. 对道路区域进行膨胀，保留紧邻道路边缘的障碍物。
3. 只保留图像下部 lower_roi_ratio 指定的区域。
4. 用该有效 ROI 与异常阈值结果求交集。

这样可以减少天空、建筑、树木和远处背景被误判为路面障碍物。

#### 1.1.4 已知目标排除

主流程先运行已知障碍物检测，再把已知目标框传入未知检测器。未知模块会按 known_box_padding_ratio 和 known_box_min_padding 扩张每个已知框，并从异常二值图中清除这些区域。

~~~text
未知候选 = 高异常像素 ∩ 道路有效区域 - 已知目标扩张区域
~~~

因此，未知检测模块的职责是补充已知检测器漏掉的异常区域，而不是再次识别车辆、行人等已知类别。

#### 1.1.5 区域生成

完成空间约束后，模块依次执行：

1. 形态学开运算，去除孤立噪点。
2. 形态学闭运算，连接相邻异常像素。
3. 提取外轮廓。
4. 根据轮廓面积占整帧面积的比例过滤过小或过大的区域。
5. 使用区域内异常分数的指定分位数作为区域置信度。
6. 为轮廓生成矩形框、面积和列优先 RLE 掩码。


### 1.2 软件架构

#### 1.2.1 类与职责

~~~text
UnknownDetector
├── 校验输入图像、道路掩码和已知目标
├── 调用 Mask2AnomalyBackend
├── 构造异常分数图
├── 应用道路 ROI 与已知框排除
├── 形态学处理和轮廓过滤
└── 返回 UnknownDetectionResult

Mask2AnomalyBackend
├── 解析项目内模型配置与权重路径
├── 校验权重 SHA-256
├── 注册 Mask2Former / Mask2Anomaly 组件
├── 构造 Detectron2 DefaultPredictor
└── 输出标准化异常分数图
~~~

#### 1.2.2 单帧数据流

~~~text
视频帧
  │
  ├── 已知检测器 ──────────────> known_objects
  │
  ├── 道路分割器 ──────────────> road_mask
  │
  └── Mask2Anomaly ───────────> anomaly_score_map
                                      │
                          road_mask + known_objects
                                      │
                                      ▼
                                未知区域后处理
                                      │
                                      ▼
                            unknown_objects / mask
                                      │
                                      ▼
                            距离估计、风险评估、可视化
~~~

#### 1.2.3 输入

UnknownDetector.predict 接收：

| 输入 | 类型 | 说明 |
|---|---|---|
| frame | BGR ndarray | 当前视频帧 |
| road_mask | ndarray | 与视频帧同尺寸的道路二值掩码 |
| known_objects | list | 已知检测器输出，可为空 |

模块只处理视频帧，不依赖摄像头标定接口。当前未知检测器本身不直接计算距离。

#### 1.2.4 输出

UnknownDetectionResult 主要字段如下：

| 字段 | 说明 |
|---|---|
| score_map | 0 到 1 的像素级异常分数图 |
| anomaly_mask | 后处理后的二值异常掩码 |
| regions | 未知区域列表 |
| inference_time_ms | 本帧未知检测耗时 |
| error_code / error_message | 统一错误信息 |
| model_version | 权重与模型版本标识 |

每个 unknown region 包含 object_id、bbox、score、area、mask_rle 和 distance。distance 初始为 None，主流程随后由 DistanceEstimator.estimate_unknown_regions 填充。

### 1.3 当前配置

默认配置位于 configs/default.yaml：

~~~yaml
unknown_detector:
  backend: "mask2anomaly"
  model:
    config_path: configs/mask2anomaly/anomaly_inference.yaml
    weights_path: checkpoints/mask2anomaly/best_contrastive_fp16.pth
    weights_sha256: 3f44ef4018beee5b4afbddc023a4eab8ce515c1a107f4b4ed4e78debccbeedfe
    device: cuda:0
    num_inlier_classes: 19
  post_processing:
    pixel_threshold: 0.5
    known_mask_threshold: 0.5
    known_box_padding_ratio: 0.1
    known_box_min_padding: 4
    min_area_ratio: 0.0002
    max_area_ratio: 0.15
    lower_roi_ratio: 0.25
    roi_dilate_kernel_size: 31
    morphology_kernel_size: 5
    region_score_quantile: 0.95
  inference:
    enable_flip_tta: false
~~~

| 参数 | 影响 |
|---|---|
| pixel_threshold | 越低越敏感，也越容易误报 |
| known_mask_threshold | 已知类别响应的抑制阈值 |
| known_box_padding_ratio | 已知框周围排除范围 |
| min_area_ratio | 过滤微小噪声 |
| max_area_ratio | 过滤覆盖画面过大的异常块 |
| lower_roi_ratio | 限制参与检测的图像下部区域 |
| roi_dilate_kernel_size | 允许检测紧邻道路边界的目标 |
| morphology_kernel_size | 控制去噪与区域连接强度 |
| region_score_quantile | 区域分数统计分位数 |
| enable_flip_tta | 水平翻转测试时增强，开启后更慢 |

配置路径均相对于项目根目录解析，所以换机器后不需要保留原电脑上的绝对 model root。只要项目内包含配置、源码和权重，并正确安装依赖即可运行。

best_contrastive_fp16.pth 是 FP16 存储版本，可减小参数文件体积。当前 Detectron2 推理图仍按框架默认精度加载和执行，不能仅凭文件名认定整条推理链路已经使用半精度计算。

### 1.4 运行结果与错误处理

#### 1.4.1 可视化表现

可视化层会在画面中绘制未知区域轮廓或矩形框，并显示 unknown 标识、异常分数和可用的距离。区域还会进入风险评估，与自车走廊发生空间关系计算。

#### 1.4.2 错误码

| 错误码 | 含义 |
|---:|---|
| 0 | 推理成功 |
| -1 | 输入或参数无效 |
| -2 | 配置、权重或模型初始化失败 |
| -3 | 推理或后处理运行失败 |

调用方应检查 error_code，而不能只根据 regions 是否为空判断失败。成功但没有发现异常时，regions 也会为空。

#### 1.4.3 测试范围

项目测试覆盖了异常分数到区域的转换、已知框排除、主流程接入和导出结构。单元测试主要验证接口与规则稳定性，不代表模型在真实道路数据上的召回率和误报率。

### 1.5 优势与局限

#### 优势

- 可以发现不在固定类别表中的道路异常物。
- 使用道路掩码约束，减少无关背景误报。
- 与已知检测结果互斥，避免同一目标重复进入风险模块。
- 输出包含分数图、区域掩码和矩形框，方便调试和后续处理。
- 模型文件与配置保存在项目内，部署不依赖原开发电脑路径。

#### 局限

- 异常检测只能说明“不像已知类别”，不能直接给出物体语义。
- 对训练域之外的光照、路面、天气和相机风格较敏感。
- Detectron2 与自定义 CUDA 算子使环境配置比普通 PyTorch 模型复杂。
- 单帧区域 ID 不提供跨帧跟踪能力。
- 阈值和面积规则需要结合目标视频重新标定。
- FP16 权重文件变小不保证推理一定更快，也可能带来精度变化。

### 1.6 使用与排查

#### 1.6.1 正常调用顺序

主流程中的正确顺序是：

1. 读取视频帧并完成质量检查。
2. 执行道路分割，得到 road_mask。
3. 执行已知目标检测，得到 known_objects。
4. 把帧、road_mask 和 known_objects 传给未知检测器。
5. 为未知区域补充距离。
6. 将已知和未知障碍物一起传给风险评估与可视化。

改变第 2 到第 4 步顺序会使道路约束或已知目标排除失效。

#### 1.6.2 Linux 环境检查

运行前至少应确认：

~~~bash
python -c 'import torch; print(torch.cuda.is_available(), torch.version.cuda)'
python -c 'import detectron2; print("detectron2 OK")'
python -c 'import torch; import MultiScaleDeformableAttention; print("MSDeformAttn OK")'
test -f checkpoints/mask2anomaly/best_contrastive_fp16.pth
~~~

导入自定义算子前先导入 torch，可避免动态库尚未加载时出现 libc10.so 找不到的问题。

#### 1.6.3 常见现象

| 现象 | 优先检查 |
|---|---|
| region_count 为 0 | 道路掩码、pixel_threshold、lower_roi_ratio |
| 已知车辆又被标成 unknown | 已知框坐标、known_box_padding_ratio、调用顺序 |
| 大片路面都是异常 | 权重、类别数、输入颜色、域差异和阈值 |
| CUDA out of memory | GPU 占用、输入分辨率、TTA 和其他模型显存 |
| 模型初始化失败 | Detectron2、自定义算子、配置路径、SHA-256 |
| 距离一直为 None | 主流程是否调用未知区域距离估计接口 |

### 1.7 后续优化建议

1. 用项目目标场景建立专门的未知障碍物验证集。
2. 分别统计像素级与区域级的召回率、误报率和延迟。
3. 加入跨帧关联，抑制只出现一帧的异常噪声。
4. 根据路面距离或透视位置使用动态面积阈值。
5. 对 FP16、混合精度、TensorRT 等方案分别做速度与精度对比。
6. 对异常区域增加可解释的语义描述，但保持与已知类别检测解耦。


---

## 2. 系统集成说明

### 2.1 主流程接口

模块在 src/main.py 中初始化并逐帧调用。模型配置来自 configs/default.yaml，模型实现位于 src/modules/unknown_detector.py，Mask2Anomaly 适配层位于 src/modules/mask2anomaly_backend.py。

### 2.2 下游依赖

| 下游模块 | 使用内容 |
|---|---|
| 距离估计 | bbox、mask_rle，为区域写入 distance |
| 风险评估 | 区域占用范围、距离、走廊空间关系 |
| 可视化 | 框、掩码、分数、距离和风险等级 |
| JSON 导出 | 区域信息、模型版本、耗时和错误状态 |

### 2.3 修改约束

- 修改 Mask2Anomaly 输出格式时，应同步更新 UnknownDetector 的适配逻辑。
- 修改 RLE 编码顺序时，应同步检查风险评估和导出端的解码。
- 修改已知目标结构时，应保留 bbox_xyxy，供排除逻辑使用。
- 新增阈值时应放入配置文件，不应写死在主流程。
- 模型加载失败应显式报错，不能静默退化为“没有未知障碍物”。

---

## 附录：关键代码位置

| 内容 | 文件 |
|---|---|
| 未知检测器与后处理 | src/modules/unknown_detector.py |
| Mask2Anomaly 适配层 | src/modules/mask2anomaly_backend.py |
| 模型推理配置 | configs/mask2anomaly/anomaly_inference.yaml |
| 系统参数 | configs/default.yaml |
| 主流程接入 | src/main.py |
| 未知区域距离估计 | src/modules/distance_estimator.py |
| 风险评估 | src/modules/risk_evaluator.py |
| 结果绘制 | src/utils/result_visualizer.py |
| 实时窗口 | src/utils/live_visualizer.py |
| 相关测试 | tests/ |
