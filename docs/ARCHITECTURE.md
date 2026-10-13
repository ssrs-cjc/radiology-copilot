# 技术架构说明

> Radiology Copilot · 科研教学原型

## 1. 总体结构

```
┌──────────────────────────────────────────────────────────────────────┐
│                          app.py (Gradio Blocks)                      │
│  工作台 │ 智能阅片 │ AI 报告 │ 教学模式 │ 科研反馈                      │
└───────┬──────────────┬──────────────┬──────────────┬─────────────────┘
        │              │              │              │
        ▼              ▼              ▼              ▼
   ┌─────────┐   ┌───────────┐  ┌──────────┐  ┌───────────┐
   │ imaging │   │segmentation│  │  agent   │  │ teaching  │
   │ 读取/   │   │ MONAI     │  │ 工具注册 │  │ 答案隐藏  │
   │ 校验/   │   │ bundle    │  │ + 调用环 │  │ + 评分    │
   │ 渲染    │   │ 推理      │  │ 路       │  │           │
   └────┬────┘   └─────┬─────┘  └────┬─────┘  └─────┬─────┘
        │              │             │              │
        │              ▼             ▼              │
        │        ┌───────────┐  ┌─────────┐         │
        │        │  metrics  │  │ report  │         │
        │        │ 体积/Dice │  │ 结构化  │         │
        │        │ 标签解码  │  │ 报告    │         │
        │        └─────┬─────┘  └────┬────┘         │
        │              │             │              │
        ▼              ▼             ▼              ▼
   ┌──────────────────────────────────────────────────────┐
   │  cases.py (统一 Case 对象)  +  storage.py (SQLite)    │
   │  NIfTI 文件 → 文件系统   ·   结构化记录 → SQLite       │
   └──────────────────────────────────────────────────────┘
                              │
                              ▼
                      research.py (反馈聚合 / CSV)
```

**核心设计**：所有模块围绕同一个 `cases.Case` 对象工作，不存在三套互相不同步的病例状态。

## 2. Agent 如何调用工具

### 2.1 职责划分

Agent **不做影像分析**。这是刻意的边界：

| 谁 | 负责什么 |
| --- | --- |
| 分割模型（SegResNet） | 从 MRI 得到肿瘤区域掩膜 |
| 确定性代码（`metrics.py`） | 由掩膜算体积、算 Dice |
| LLM Agent | 决定调用哪个工具、按什么顺序、把结构化结果组织成可读回答 |

### 2.2 工具注册表

`agent.py` 中的 `ToolRegistry` 持有全部可达工具，每个工具声明名字、说明、JSON Schema 参数与实现函数：

| 工具 | 作用 | 状态语义 |
| --- | --- | --- |
| `get_case_info` | 病例基本信息 | 无病例 → `error` |
| `validate_mri` | 输入检查结果 | 检查未通过 → `error` |
| `run_segmentation` | 运行分割 | **永远 `refused`**（见下） |
| `get_segmentation_results` | 分割结果与模型版本 | 无结果 → `error` |
| `calculate_volume` | 读取已算好的体积 | 无掩膜 → `error` |
| `get_review_status` | 报告审核状态 | — |
| `get_model_evaluation` | 与参考标注的 Dice | 无参考标注 → 明确说明 |
| `generate_report_draft` | 报告状态 | 未确认 → `refused` |
| `get_research_feedback` | 反馈统计 | — |

**每次调用都返回 `ok` / `error` / `refused` 三态之一，并被记录进 trace。**
不存在"模型说它调用了工具"但工具没跑的情况。

### 2.3 需要人工确认的操作

`run_segmentation` 与 `generate_report_draft` 被显式设计为 **Agent 无法自行触发**：

```python
def run_segmentation(confirm: bool = False) -> ToolResult:
    if not confirm:
        return ToolResult(..., "refused", "运行分割模型需要用户明确确认", ...)
    return ToolResult(..., "refused", "请在界面上点击「运行 AI 分割」按钮以执行模型", ...)
```

即使模型自己传 `confirm=true` 也会被拒绝——确认必须来自 UI，不能来自模型自己的参数。
这防止了 LLM 绕过用户直接消耗算力或改变系统状态。

**没有暴露任何 shell、文件写入或删除工具。**

### 2.4 两种运行模式（都如实标注）

| 模式 | 触发条件 | 行为 |
| --- | --- | --- |
| `llm_tool_calling` | API 已配置 **且** 通过工具调用能力探测 | LLM 真实选择并调用工具，多轮循环，结果回灌 |
| `fixed_workflow` | 未配置 API，或探测失败 | 关键词路由到确定性 Python 工具链，结果直接汇总 |

**为什么需要"探测"**：部分 OpenAI 兼容服务会静默忽略 `tools` 参数。若不探测，
模型会**叙述**工具调用却没有真正执行——这正是本项目必须避免的失败模式。
`probe_tool_calling()` 发一次真实请求，检查返回里是否真的有 `tool_calls`。

界面始终显示当前模式，API 不可用时**不会**伪装成成功调用大模型。

## 3. 医学影像模型如何集成

### 3.1 模型来源

MONAI Model Zoo 官方 bundle：`MONAI/brats_mri_segmentation`（SegResNet，v0.5.4）。

**架构不硬编码**：从 bundle 自带的 `configs/inference.json` 中读出 `network_def`，
交给 MONAI `ConfigParser` 实例化——构造出来的对象必然就是权重被训练的那个结构。

```python
cfg = json.loads((bundle_dir / "configs/inference.json").read_text())
network = ConfigParser(config=cfg).get_parsed_content("network_def")
network.load_state_dict(torch.load(weights), strict=True)   # 不完整加载直接报错
```

`strict=True` 是刻意的：部分加载会留下随机初始化的层，而它照样会输出看起来很自信的掩膜。

### 3.2 从 bundle 元数据读出的真实规格

以下全部来自 bundle 自带配置，**不是来自记忆或猜测**：

| 项目 | 值 | 来源 |
| --- | --- | --- |
| 输入通道 | `T1c, T1, T2, FLAIR` | `configs/metadata.json → network_data_format.inputs.channel_def` |
| 输出通道 | `0=TC, 1=WT, 2=ET` | 同上 `outputs.channel_def`（注意 **WT 是通道 1**） |
| 预处理 | `NormalizeIntensityd(nonzero=True, channel_wise=True)` | `configs/inference.json` |
| 推理 | `SlidingWindowInferer(roi=[240,240,160], batch=1, overlap=0.5)` | 同上 |
| 后处理 | sigmoid → 阈值 0.5 → `ET(4) > TC(1) > WT(2)` 编码 | 同上 |
| 通道顺序验证 | `"image": t1ce + t1 + t2 + flair` | `scripts/prepare_datalist.py` |

### 3.3 一个真实的踩坑记录：归一化

最初按文档手写了逐通道零均值标准化，**对整个体积**做 `(x-μ)/σ`。
BraTS 是去颅骨数据，背景全为 0，于是背景被映射到约 **−4σ**——
一个网络训练时从未见过的输入分布。

后果：WT Dice 只有 **0.397**，预测区域几乎是参考的 1/4。

对照 MONAI `NormalizeIntensity._normalize` 源码后发现：

```python
if slices is not None:
    img[slices] = (masked_img - _sub) / _div     # 只对非零体素做归一化
```

**背景必须保持为 0。** 改为直接调用 MONAI 自己的 `NormalizeIntensity`
（而不是继续手写）后，WT Dice 从 **0.397 → 0.9255**。

> 教训：预处理必须调用官方实现，不能凭文档描述手写；一次语义偏差足以让模型"看起来很差"。

### 3.4 标签编码：按数据集声明 + 实测校验

不同数据集的整数标签含义不同，猜错会让所有 Dice 失去意义：

| 数据集 | WT | TC | ET |
| --- | --- | --- | --- |
| BraTS 2018–2021 | 1, 2, 4 | 1, 4 | **4** |
| MSD Task01_BrainTumour | 1, 2, 3 | 2, 3 | **3** |

同一个整数 `3` 在两个数据集里含义完全不同。因此 `cases.py` 中每个数据集适配器
**显式声明**其约定，`metrics.verify_convention()` 在运行时**对照真实标签值校验**，
不一致就报告而不是静默继续。

本项目实测 BraTS 2020 数据的标签为 `{0, 1, 2, 4}`——**没有标签 3**，与 BraTS 约定一致。

### 3.5 嵌套区域的正确处理

三个输出通道**不是互斥标签**，而是嵌套区域：`ET ⊆ TC ⊆ WT`。

错误的做法是把三个通道当成三个独立类别。正确的做法是先编码成 BraTS 风格整数标签图
（优先级 `ET > TC > WT`），再由标签图反推三个区域掩膜——这样嵌套关系在结构上不可能错。

模型输出偶尔仍会出现少量违反嵌套的体素（实测某病例 46 个体素）。系统**如实报告**
这一现象并按优先级编码，而不是悄悄掩盖。

### 3.6 推理设备选择

这是一台共享 GPU 服务器。`select_device()` 通过 NVML 读取每张卡的**空闲显存与利用率**，
优先选择"空闲显存充足 **且** 利用率低"的卡，并支持 `RADIOLOGY_DEVICE` 覆盖：

- 显存低于 6 GB 的卡直接排除，避免影响他人任务
- 全部卡都不合适时**降级到 CPU**，而不是去挤占
- 界面与日志如实显示所选设备及原因

## 4. 数据如何在诊、教、研之间传递

### 4.1 统一 Case 对象

```python
Case(
    case_id, dataset_source, modalities, image_paths,
    reference_mask_path, prediction_mask_path, corrected_mask_path,
    model_name, model_version, segmentation_status,
    measurements, report_draft, review_status,
    teaching_record, research_feedback,
    label_convention, data_origin,       # 来源与编码约定，UI 必须可见
)
```

### 4.2 存储分层

| 内容 | 存放位置 | 原因 |
| --- | --- | --- |
| NIfTI 体数据（影像、掩膜） | `data/`、`outputs/predictions/` 文件系统 | 体数据不适合进数据库 |
| 结构化记录（病例、分析、报告、教学、反馈、Agent trace） | `outputs/radiology_copilot.db` (SQLite) | 需要查询与聚合 |
| 标量结果（体积、Dice） | SQLite，随记录一起 | 仪表盘直接聚合，无需重算 |

### 4.3 三类掩膜严格区分

同一病例可能存在三份掩膜，**永不互相覆盖**：

1. `reference_mask_path` — 数据集专家标注（只读）
2. `prediction_mask_path` — 模型预测（`outputs/predictions/<case>_pred.nii.gz`）
3. `corrected_mask_path` — 人工修正版本（预留）

UI 与报告始终标注当前显示的是哪一份。体积与 Dice 都带有 `source` 字段。

### 4.4 审核后的病例保存独立版本

`reports` 表是**追加式**的：每次保存产生新的 `revision`，审核产生带 `approved_at` 的新行。
`analysis_runs` 同样是追加式的，失败运行也会记录（`status='failed'`），
因此可以追溯"这个病例什么时候跑过哪个模型版本、结果如何"。

审核操作**不会**删除或覆盖原始模型预测。

## 5. 系统如何保存科研反馈

```
analysis_runs (模型评测, 含 evaluation JSON)
        │
        ├──→ research.aggregate() ──→ Dashboard（平均 Dice / 标准差 / 困难病例）
        │
research_feedback (人工确认的错误分类 + 备注)
        │
        └──→ research.feedback_csv() ──→ CSV 导出
```

**两条准确性规则**：

1. **只聚合真实测量值**。没有参考标注的病例对 Dice 统计**贡献为零**，
   而不是贡献一个 0 分——后者会把平均值拉低成看起来像模型很差的假象。
2. **空是合法状态**。没有评测数据时显示空状态说明，而不是显示 0。

错误分类由**人工确认**后写入；系统只根据体素级误差方向提供预选建议
（`suggest_error_type`），不自动写入。

## 6. 测试策略

`tests/` 共 **311** 个测试，全部可在**没有数据集、没有模型权重**的机器上通过：

| 文件 | 覆盖 |
| --- | --- |
| `test_metrics.py` | 标签约定、区域嵌套、体积、Dice 边界情形（双空/单空/形状不符） |
| `test_cases_imaging.py` | 通道顺序映射、病例完整性、输入校验、几何、渲染窗宽 |
| `test_report_teaching.py` | 报告数值一致性、不推断病理、教学答案隐藏、掩膜评分 |
| `test_agent_storage.py` | 工具三态、工具拒绝、无 shell 工具、两种模式、持久化、聚合 |
| `test_corrections.py` | 修正掩膜导入、空间匹配、版本化、来源标注、跨病例隔离 |
| `test_ux_fixes.py` | 阅片构图（裁剪/等比/掩膜最近邻）、教学答案隔离、统计口径、误差可视化、跨病例隔离 |
| `test_agent_llm.py` | 工具参数校验、调用失败反馈、API 未配置降级、非法工具名、未授权状态修改、报告数值一致性、trace 真实状态与耗时、跨病例隔离 |
| `test_app_callbacks.py` | 逐个调用全部 UI 回调（含修正回流），防止回调内的静默错误 |
| `test_agent_llm.py`（本地部分） | provider 选择、自动降级、文本态 `<tool_call>` 解析、思考块剥离、适配器不放宽校验 |

其中若干测试是针对**实际发生过的 bug** 写的回归测试（如窗口百分位把背景计入导致
影像全白、非零归一化缺失导致 Dice 崩塌）。

运行：

```bash
python -m pytest tests/ -v
```

## 7. 关键取舍

| 决定 | 理由 |
| --- | --- |
| Gradio 单进程，不用前后端分离 | v1 目标是 `python app.py` 即可运行；交付速度优先 |
| SQLite 而非 Postgres/Redis | 单用户原型，零运维成本 |
| 单 Agent + 工具注册表，不做多 Agent | 任务本身不需要多 Agent，引入只会增加不可观测性 |
| 体积/Dice 全部确定性计算 | 医学数值不允许 LLM 参与运算 |
| 掩膜图像与结构化解耦存储 | 体数据进数据库会让备份与迁移都变难 |
