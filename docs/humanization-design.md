# SoulX-Podcast 拟人化文本预处理器 — 设计文档

> 版本: v2 | 日期: 2026-07-07 | 文件: `chengbook_tts/engines/soulxpodcast_humanize.py`

---

## 1. 总体架构

### 1.1 设计理念

**纯文本预处理，不碰模型推理。** 所有拟人化操作在文本送入 TTS 模型之前完成——修改的是 tokenizer 输入字符串本身，不改变模型权重、计算图或采样逻辑。这样做的核心优势：

- **模型无关性**：只要 tokenizer 能识别注入的特殊 token，任何 LLM-based TTS 都能复用这套逻辑
- **零推理开销**：预处理耗时 < 1ms，不影响合成速度
- **安全可逆**：出问题只需关掉开关 `humanize=False`，不影响正常合成

### 1.2 处理流水线

```
输入文本
  │
  ├─[1] 句首话语标记注入    _inject_discourse_markers()
  ├─[2] 副语言标签注入      _inject_paralinguistic_tags()
  ├─[3] 自然停顿标记注入    _inject_natural_pauses()
  ├─[4] 拟人化瑕疵注入      _apply_imperfections()
  ├─[5] 句末语气词注入      _inject_sentence_final_particles()
  ├─[6] 后处理清理          _cleanup_unnatural_combinations()
  │
  └─→ 送入 TTS tokenizer → 合成
```

**流水线顺序是经过设计的：**

1. 先做**句子级**注入（话语标记、副语言标签）——它们依赖原始文本的句子边界，不能受后续改动干扰
2. 再做**从句级**注入（停顿、瑕疵、语气词）——它们在已有结构上附加细节
3. 最后做**清理**——修复不同步骤独立运行产生的非自然相邻组合

每一步的输出是下一步的输入，每一步用 `random.seed()` 重新播种（每次合成调用结果不同）。

### 1.3 依赖

- `re` — 正则匹配中文标点边界
- `random` — 所有概率决策（`random.random()`, `random.choice()`, `random.choices()`）
- `dataclasses` — 配置对象

不依赖 torch、transformers 或任何 ML 库。

---

## 2. 五大拟人化策略详解

### 2.1 策略一：副语言标签注入

#### 目标
模拟真人说话时的呼吸、叹气、咳嗽、清嗓子等非语言声音。

#### 原理
SoulX-Podcast 使用 Qwen3 tokenizer，原生训练时包含 5 种特殊 token：

| 标签 | 含义 | 权重 | 说明 |
|------|------|------|------|
| `<\|breathing\|>` | 呼吸声 | 8 | 最自然，频率最高 |
| `<\|sigh\|>` | 叹气 | 3 | 失望/疲惫时自然出现 |
| `<\|coughing\|>` | 咳嗽 | 2 | 偶发 |
| `<\|throat_clearing\|>` | 清嗓子 | 1 | 较少见 |
| ~~`<\|laughter\|>`~~ | 轻笑 | — | 已移除，合成效果不自然 |

模型在预训练时见过这些 token，知道它们对应的声学特征。将标签插入文本后，模型会在合成时自动生成对应的非语言声音。

#### 插入位置
在**自然停顿点之后**（`_TAG_BREAK_RE`：`。！？，、；：\n!?,:;`），且仅在中文上下文（±10 字符内有中文字符）中。

#### 核心算法
```python
# 1. 找到所有标点后的可插入位置（反向记录）
# 2. 对每个位置，按 tag_probability 决定是否插入
# 3. 按权重加权随机选择标签类型
tag = random.choices(tags, weights=normalized_weights, k=1)[0]
text = text[:pos] + tag + text[pos:]  # 逆向遍历，索引不偏移
```

#### 权重归一化
权重 (8, 3, 2, 1) → 总权重 14 → 归一化概率 (57%, 21%, 14%, 7%)。修改权重无需改概率加和逻辑，自动归一化。

---

### 2.2 策略二：自然停顿标记

#### 目标
在逗号/分号处插入省略号（`……`、`…`、`...`）或破折号（`——`），让模型合成出更长的停顿间隙，模拟真人换气和思考的节奏。

#### 插入位置
仅在**中文逗号/分号**（`，、；：`）**之前**插入，不在英文标点处插入。

```python
# 找到所有中文逗号位置 → 以 pause_probability 随机决定是否插入
text = text[:pos] + marker + text[pos:]  # marker 在逗号之前："...，"
```

#### 标记选择
随机从 `['……', '…', '...', '——']` 中选取。不同的省略号长度给模型略微不同的停顿信号。

---

### 2.3 策略三：拟人化瑕疵

#### 目标
模拟真人口语中的不完美——填充词（思考缓冲）、口吃重复（卡顿）、自我纠错（犹豫）。这是让语音听起来"像真人"而非"像播报"的核心策略。

#### 三种瑕疵类型

**a) 填充词 (filled pauses) — 占比 60%**
```
嗯，| 啊，| 呃，| 那个，| 就是说，| 然后呢，| 就，| 怎么说呢，| 就是，
```
模拟思考中的停顿缓冲。全部以逗号结尾，确保与后续内容有自然断开。

**b) 口吃/重复 (stutters) — 占比 20%**
```
这个这个 | 就就 | 然后然后 | 那那 | 这这 | 我我 | 是是 | 在在 | 要要
```
模拟说话卡顿、单字/双字重复。**注意：口吃不放句末**（详见 §2.3.1）。

**c) 纠错/犹豫 (interjections) — 占比 20%**
```
呃，| 哎，| 哦，| 呃等一下，| 不对，应该是呃，| 嗯...不是，| 啊说错了，| 其实，
```
模拟说错话然后自我纠正的自然行为。

#### 插入位置
在**句中停顿**（`_MID_SENTENCE_PUNCT = {'，', '、', '；', '：'}`）前的从句边界。不在句末标点（。！？）前插入，不在文本末尾插入。

```python
# _CN_CLAUSE_RE.split() → [clause0, delim0, clause1, delim1, ...]
# 偶数索引 = 内容从句，奇数索引 = 分隔符
# 只在偶数索引（内容从句）后、分隔符为句中停顿时插入
```

#### 2.3.1 口吃句末抑制规则

**问题**：口吃重复放在句末不自然。例如："说实话挺失望的这个这个，毕竟..." → "这个这个" 在句末前最后一个逗号处。

**规则**：生成口吃后，检查**下一个**分隔符（`clauses[i+3]`）是否为句末标点。若是，跳过本次插入。
```python
if imperfection in _STUTTERS:
    if clauses[i + 3] in _SENTENCE_END_PUNCT:
        continue  # 跳过
```

---

### 2.4 策略四：句末语气词

#### 目标
在陈述句末添加口语化语气词，让书面语朗读听起来像随口聊天而非念稿。

#### 语气词库
```
吧 | 嘛 | 啊 | 呢 | 呀 | 呗
```
（`哈`、`哦` 已移除——不符合中文口语习惯；`呀` 保留——常见于轻松语境）

#### 插入位置
在**句末标点**（`。！？.!?`）**之前**，仅当前 6 个字符内有中文时生效：
```python
text = text[:pos] + particle + text[pos:]  # "进展。" → "进展呀。"
```

---

### 2.5 策略五：句首话语标记

#### 目标
在句子开头插入思考停顿标记，模拟开口前的短暂犹豫——真人说话时很少直接开始，通常会有一个"嗯..."或"那个..."的缓冲。

#### 标记库
```
嗯... | 呃... | 那个... | 这个... | 就是说... | 哎，| 哦，
```

#### 插入位置
仅在**句末标点后**的新句子开头，且新句子以中文字符开头时生效：
```python
# _SENTENCE_END_RE = re.compile(r'([。！？.!?\n])')
# split 后的偶数索引（i>0, i%2==0）是标点后的新句子开头
```

---

## 3. SamplingParams 扰动

除了文本层面的注入，还通过随机抖动采样参数来模拟语速/语调的自然波动：

| 参数 | 抖动范围 | 下限 | 上限 | 作用 |
|------|----------|------|------|------|
| `temperature` | ±0.10 | 0.1 | 2.0 | 控制随机性，高→更多样但可能不稳定 |
| `top_k` | ±20 | 1 | — | 控制候选 token 数 |
| `top_p` | ±0.08 | 0.1 | 1.0 | nucleus sampling 阈值 |
| `repetition_penalty` | ±0.08 | 0.5 | 2.0 | 重复惩罚 |
| `tau_r` | ±0.10 | 0.02 | 0.6 | RAS 语速控制（当前 `use_ras=False` 下不生效） |

**注意**：当前 `use_ras=False`，`tau_r` 和 `win_size` 的抖动实际上不参与推理。这是为了兼容新版 transformers（移除了 `custom_generate` kwarg）。如需启用 RAS 语速控制，需要将 RAS 逻辑适配为 `LogitsProcessor` 形式。

---

## 4. 后处理清理

由于 5 个注入步骤独立运行（每步不知道其他步骤做了什么），可能在相邻位置产生非自然组合。清理步骤在流水线最后统一修复。

### 规则表

| # | 模式 | 示例 | 修复 | 正则 |
|---|------|------|------|------|
| 1 | 填充词干+语气词粘连 | `嗯呀` | `嗯，呀` | `([嗯呃啊])([吧嘛呢呀呗])` → `\1，\2` |
| 2 | 连续逗号 | `呃，，` | `呃，` | `，{2,}` → `，` |
| 3 | 语气词互斥 | `吧嘛` | `吧` | `([吧嘛啊呢呀呗])([吧嘛啊呢呀呗])` → `\1` |
| 4 | 句号后语气词 | `.嘛` | `.` | `([。！？.!?])([吧嘛啊呢呀呗])` → `\1` |
| 5 | 语气词间夹标点再粘语气词 | `呀.嘛` | `呀.` | `([吧嘛啊呢呀呗])[.…]+([吧嘛啊呢呀呗])` → `\1.` |

---

## 5. 配置系统

### 5.1 HumanizeConfig

```python
@dataclass
class HumanizeConfig:
    enabled: bool = False              # 总开关

    # 五大策略开关 + 概率
    inject_tags: bool = True           # 副语言标签
    tag_probability: float = 0.15      #   每个停顿点插入标签的概率

    inject_pauses: bool = True         # 自然停顿
    pause_probability: float = 0.10    #   每个逗号处插入停顿的概率

    add_imperfections: bool = True     # 说话瑕疵
    imperfection_probability: float = 0.08  # 每个从句边界插入瑕疵的概率
    filled_pause_rate: float = 0.60    #   瑕疵中填充词的比例
    stutter_rate: float = 0.20         #   瑕疵中口吃的比例
    interjection_rate: float = 0.20    #   瑕疵中纠错的比例

    add_sentence_final: bool = True    # 句末语气词
    sentence_final_probability: float = 0.15

    add_discourse_markers: bool = True # 句首话语标记
    discourse_marker_probability: float = 0.12

    perturb_sampling: bool = True      # 采样参数扰动
    perturb_config: HumanizePerturbConfig = field(...)
```

### 5.2 预设强度

| 参数 | light | moderate | heavy |
|------|-------|----------|-------|
| tag_probability | 0.15 | 0.30 | 0.40 |
| pause_probability | 0.10 | 0.20 | 0.35 |
| imperfection_probability | 0.08 | 0.18 | 0.30 |
| sentence_final_probability | 0.15 | 0.30 | 0.45 |
| discourse_marker_probability | 0.12 | 0.25 | 0.40 |

**调参经验**：概率不是越高越好。heavy 级别在短文本（< 30 字）上可能产生过于密集的注入，让语音听起来支离破碎。建议：
- 短文本（< 30 字）→ light
- 中等文本（30-100 字）→ moderate
- 长文本（> 100 字）→ moderate 或 heavy

---

## 6. 引擎集成

### 6.1 调用点

在 [soulxpodcast.py](chengbook_tts/engines/soulxpodcast.py) 的 `synthesize()` 方法中：

```python
# 1. 构建配置
humanize_config = self._build_humanize_config(kwargs)
# 2. 文本预处理
text = apply_humanization(text, humanize_config)
# 3. 采样参数扰动
sampling_params = perturb_sampling_params(sampling_params, humanize_config)
# 4. 后续正常的 tokenize + 模型推理
```

### 6.2 API 接口

通过 `TTSRequest` 的两个字段控制：
- `humanize: bool` — 是否启用
- `humanize_level: str` — 强度等级 (`'light'` / `'moderate'` / `'heavy'`)

```python
def _build_humanize_config(self, kwargs):
    humanize = kwargs.get('humanize', False)
    level = kwargs.get('humanize_level', 'moderate')
    return build_config(humanize, level)
```

### 6.3 调试日志

启用后每次合成都打印预处理前后的文本对比：
```
[Humanize] ★ 预处理前: 您好，我有个订单问题想反馈一下。
[Humanize] ★ 预处理后: 您好…，<|breathing|>我有个订单问题想反馈一下呀。
```

---

## 7. 移植到其他模型

### 7.1 模型必须满足的条件

| 条件 | 说明 |
|------|------|
| LLM-based TTS | 基于 token→speech 的架构，不是纯声学模型 |
| 中文 tokenizer | tokenizer 词表覆盖中文 + 特殊标签 token |
| 支持特殊 token | 模型训练数据中包含过 `<\|breathing\|>` 等标签，或至少 tokenizer 能正确 encode 这些 token（不会拆成子词碎片） |

### 7.2 移植步骤

**Step 1: 确定标签支持**
检查目标模型的 tokenizer 是否能识别副语言标签：
```python
tokenizer.encode('<|breathing|>')  # 应为单个 token id，而非拆成多个子词
```
如果模型没训练过这些标签，合成时可能产生噪音或跳过。此时应：
- 禁用 `inject_tags`（`tag_probability=0` 或 `inject_tags=False`）
- 保留文本层面的人性化策略（停顿、瑕疵、语气词）——这些纯文本注入任何中文 TTS 都能受益

**Step 2: 调整正则表达式**
如果目标语言不是中文，需替换：
- `_CN_CLAUSE_RE` → 目标语言的从句边界规则
- `_has_chinese()` → 目标语言的字符范围检测
- `_MID_SENTENCE_PUNCT` / `_SENTENCE_END_PUNCT` → 目标语言的标点体系
- 所有瑕疵模板（`_FILLED_PAUSES`, `_STUTTERS` 等）→ 目标语言的口语习惯

**Step 3: 调整注入位置逻辑**
不同语言的句法结构不同。例如英文：
- 副语言标签插入位置：逗号、句号、分号后
- 句末语气词不适用（英文无此习惯）→ 替换为 `you know`, `right?` 等话语标记
- 口吃模板：`I-I`, `it's it's`, `you know, like,`

**Step 4: 移除 SamplingParams 扰动**
`perturb_sampling_params()` 依赖 `soulxpodcast.config.SamplingParams` 类型。移植时要么：
- 替换为目标的采样参数类型
- 或直接设为 `perturb_sampling=False`

### 7.3 最小化移植（只保留文本策略）

如果只想快速试用文本层面的拟人化（不依赖任何模型特定标签），最小配置：

```python
config = HumanizeConfig(
    enabled=True,
    inject_tags=False,          # 关掉标签注入
    inject_pauses=True,         # 保留停顿
    add_imperfections=True,     # 保留瑕疵
    add_sentence_final=True,    # 保留语气词
    add_discourse_markers=True, # 保留话语标记
    perturb_sampling=False,     # 关掉参数扰动
)
```

这只需要改文本字符串，对任何中文 TTS 模型都安全。

---

## 8. 已知限制与改进方向

### 8.1 当前限制

1. **中文专用**：所有正则、模板、语气词均为中文设计。英文/日文等需重写模板。
2. **无上下文感知**：每句独立随机注入，不考虑前后句的注入密度。可能出现连续三句都有句末语气词（不自然）。
3. **无文本语义理解**：不会根据语义内容调整策略——表达悲伤的内容和表达开心的内容注入相同的标签。
4. **RAS 语速控制未启用**：`use_ras=False`，`tau_r` 扰动不生效。原因为新版 transformers 的 `custom_generate` 兼容性问题。
5. **单次调用内概率独立**：同一段文本内，每个注入点的概率判断互相独立，不做全局密度控制。

### 8.2 改进方向

- **全局密度限制**：统计已注入数量，超出阈值后降低后续概率
- **情感适配**：根据 `emotion` 参数调整注入策略（sad → 更多叹气、更慢停顿；happy → 少叹气、多语气词）
- **文本长度自适应**：短文本自动降级到 light，长文本允许 moderate/heavy
- **RAS LogitsProcessor**：将 RAS 逻辑改写为 `transformers.LogitsProcessor` 子类，恢复 `use_ras=True` 支持
- **注入位置可视化**：在调试日志中标注每个注入点的策略类型，方便调参

---

## 9. 文件清单

| 文件 | 作用 |
|------|------|
| `chengbook_tts/engines/soulxpodcast_humanize.py` | 拟人化核心实现（本文档描述的全部内容） |
| `chengbook_tts/engines/soulxpodcast.py` | 引擎适配器，在 `synthesize()` 中调用 humanize |
| `chengbook_tts/server/models.py` | API 模型 `TTSRequest`（`humanize` + `humanize_level` 字段） |
| `chengbook_tts/server/routes/native.py` | 路由层，传递 humanize kwargs 到引擎 |
| `chengbook_tts/server/routes/profile.py` | 用户配置持久化（`humanize` + `humanizeLevel`） |
| `chengbook_tts/config/settings.py` | 默认配置 `HUMANIZE_ENABLED` / `HUMANIZE_LEVEL` |
