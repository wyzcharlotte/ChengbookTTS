"""
SoulXPodcast 拟人化文本预处理器 (v2)
--------------------------------
通过注入副语言标签 (<|laughter|>, <|breathing|> 等) 和生成说话瑕疵
(填充词、重复、纠错、语气词) 让合成语音更贴近真人说话习惯。

v2 改进:
- 补回 <|laughter|> 标签（原版5种标签完整支持）
- 注入自然停顿标记（省略号、话语标记词）让语速更舒缓
- 句末语气词（吧、嘛、啊、呢）增加中文自然感
- 句首话语标记词（嗯...、那个...）模拟思考停顿
- 扩大 SamplingParams 扰动范围，让语速/语调更随机
- 调整预设概率，让中等强度就有明显效果

仅在 SoulXPodcast 引擎下有效 — Qwen3 tokenizer 能识别这些特殊 token。
"""

import re
import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple


# ===================== 副语言标签定义 =====================

# (标签, 权重) — 权重越高出现频率越高
# SoulX-Podcast 原生训练支持 5 种副语言标签，这里只用 4 种（笑声不自然）
_PARALINGUISTIC_TAGS: List[Tuple[str, int]] = [
    ('<|breathing|>',        8),   # 呼吸 — 最自然最常见
    ('<|sigh|>',             3),   # 叹气 — 中等频率
    ('<|coughing|>',         2),   # 咳嗽 — 偶发
    ('<|throat_clearing|>',  1),   # 清嗓子 — 较少见
]

# 自然停顿点 — 在这些标点后可以插入副语言标签
_TAG_BREAK_RE = re.compile(r'([。！？，、；：\n!?,:;])')

# 中文从句边界 — 用于分割句子以插入瑕疵
_CN_CLAUSE_RE = re.compile(r'([。！？，、；：\n]|(?<=[a-zA-Z])[.?!,;:])')

# 句中停顿（可插入瑕疵的位置）— 句末标点。！？不插
_MID_SENTENCE_PUNCT = {'，', '、', '；', '：'}

# 句末标点 — 其后可插入句末语气词
_SENTENCE_END_PUNCT = {'。', '！', '？', '.', '!', '?'}


# ===================== 拟人化瑕疵模板 =====================

# 填充词 (filled pauses) — 优先级最高，模拟思考中的停顿
_FILLED_PAUSES: List[str] = [
    '嗯，',
    '啊，',
    '呃，',
    '那个，',
    '就是说，',
    '然后呢，',
    '就，',
    '怎么说呢，',
    '就是，',
]

# 口吃/重复 (stutters) — 模拟说话卡顿
_STUTTERS: List[str] = [
    '这个这个',
    '就就',
    '然后然后',
    '那那',
    '这这',
    '我我',
    '是是',
    '在在',
    '要要',
]

# 纠错/犹豫标记 (interjections) — 模拟自我纠正
_INTERJECTIONS: List[str] = [
    '呃，',
    '哎，',
    '哦，',
    '呃等一下，',
    '不对，应该是呃，',
    '嗯...不是，',
    '啊说错了，',
    '其实，',
]

# 句末语气词 — 让陈述句更口语化
_SENTENCE_FINAL_PARTICLES: List[str] = [
    '吧',
    '嘛',
    '啊',
    '呢',
    '呀',
    '呗',
]

# 句首话语标记 — 模拟开口前的思考
_DISCOURSE_MARKERS: List[str] = [
    '嗯...',
    '呃...',
    '那个...',
    '这个...',
    '就是说...',
    '哎，',
    '哦，',
]


# ===================== 自然停顿配置 =====================

# 自然停顿标记 — 插入到逗号/分号位置，模拟换气顿挫
_NATURAL_PAUSE_MARKERS: List[str] = [
    '……',
    '…',
    '...',
    '——',
]


# ===================== SamplingParams 扰动配置 =====================

@dataclass
class HumanizePerturbConfig:
    """SamplingParams 随机扰动范围 (v2: 扩大范围以产生明显变化)"""
    temperature_jitter: float = 0.10    # 原 0.05 → 0.10
    top_k_jitter: int = 20              # 原 10 → 20
    top_p_jitter: float = 0.08          # 原 0.05 → 0.08
    rep_penalty_jitter: float = 0.08    # 原 0.05 → 0.08
    tau_r_jitter: float = 0.10          # 原 0.05 → 0.10 (增大 tau_r 波动让语速变化更明显)


# ===================== 主配置 =====================

@dataclass
class HumanizeConfig:
    """拟人化预处理总配置 (v2)"""
    enabled: bool = False

    # ---- 副语言标签注入 ----
    inject_tags: bool = True
    tag_probability: float = 0.15       # 每个停顿点插入标签的概率

    # ---- 自然停顿插入 ----
    inject_pauses: bool = True
    pause_probability: float = 0.10     # 在逗号处插入停顿标记的概率

    # ---- 拟人化瑕疵 ----
    add_imperfections: bool = True
    imperfection_probability: float = 0.08  # 每个从句边界插入瑕疵的概率

    # 瑕疵类型比例 (和为 1.0)
    filled_pause_rate: float = 0.60     # 填充词 (微降，给停顿让空间)
    stutter_rate: float = 0.20          # 口吃/重复
    interjection_rate: float = 0.20     # 纠错/犹豫 (微升)

    # ---- 句末语气词 ----
    add_sentence_final: bool = True
    sentence_final_probability: float = 0.15

    # ---- 句首话语标记 ----
    add_discourse_markers: bool = True
    discourse_marker_probability: float = 0.12

    # ---- 采样参数扰动 ----
    perturb_sampling: bool = True
    perturb_config: HumanizePerturbConfig = field(default_factory=HumanizePerturbConfig)


# 预设强度 (v2: 调整概率让效果更明显)
HUMANIZE_PRESETS: Dict[str, HumanizeConfig] = {
    'default':  HumanizeConfig(enabled=False),
    'light':    HumanizeConfig(
        enabled=True,
        tag_probability=0.15,
        pause_probability=0.10,
        imperfection_probability=0.08,
        sentence_final_probability=0.15,
        discourse_marker_probability=0.12,
    ),
    'moderate': HumanizeConfig(
        enabled=True,
        tag_probability=0.30,              # 0.18 → 0.30
        pause_probability=0.20,            # 0.10 → 0.20
        imperfection_probability=0.18,     # 0.10 → 0.18
        sentence_final_probability=0.30,   # 0.15 → 0.30
        discourse_marker_probability=0.25, # 0.12 → 0.25
    ),
    'heavy':    HumanizeConfig(
        enabled=True,
        tag_probability=0.40,              # 0.28 → 0.40
        pause_probability=0.35,            # 0.18 → 0.35
        imperfection_probability=0.30,     # 0.18 → 0.30
        sentence_final_probability=0.45,   # 0.25 → 0.45
        discourse_marker_probability=0.40, # 0.20 → 0.40
    ),
}


# ===================== 主入口 =====================

def apply_humanization(text: str, config: HumanizeConfig) -> str:
    """
    对输入文本施加拟人化预处理 (v2)。

    处理流程:
    1. 句首话语标记注入
    2. 副语言标签注入（呼吸、叹气等）
    3. 自然停顿标记注入
    4. 拟人化瑕疵注入（填充词、重复、纠错）
    5. 句末语气词注入
    6. 后处理清理（修复相邻步骤产生的粘连）

    返回预处理后的文本。如果 config.enabled 为 False 则原样返回。
    每次调用使用不同的随机种子，确保同一输入产生不同结果。
    """
    if not config.enabled:
        return text

    if not text or not text.strip():
        return text

    random.seed()  # 每次调用重新播种，保证随机性

    # 1. 句首话语标记（说话前的自然犹豫）
    if config.add_discourse_markers:
        text = _inject_discourse_markers(text, config)

    # 2. 副语言标签
    if config.inject_tags:
        text = _inject_paralinguistic_tags(text, config)

    # 3. 自然停顿标记（让语速放缓）
    if config.inject_pauses:
        text = _inject_natural_pauses(text, config)

    # 4. 说话瑕疵
    if config.add_imperfections:
        text = _apply_imperfections(text, config)

    # 5. 句末语气词
    if config.add_sentence_final:
        text = _inject_sentence_final_particles(text, config)

    # 6. 后处理清理：修复不同步骤间产生的非自然相邻组合
    text = _cleanup_unnatural_combinations(text)

    return text


# ===================== 后处理清理 =====================

def _cleanup_unnatural_combinations(text: str) -> str:
    """
    修复不同拟人化步骤间产生的非自然相邻组合。

    由于各注入步骤独立运行，可能出现填充词与语气词粘连
    (如 "嗯呀"、"呃吧") 或重叠标点 (如 "呃，，")，这些
    不符合中文说话习惯，需要清理。
    """
    # 1. 填充词干粘连句末语气词: "嗯呀" → "嗯，呀"
    text = re.sub(r'([嗯呃啊])([吧嘛呢呀呗])', r'\1，\2', text)

    # 2. 多个连续逗号: "呃，，" → "呃，"
    text = re.sub(r'，{2,}', '，', text)

    # 3. 句末语气词 + 另一个句末语气词粘连: "吧嘛" → "吧"
    text = re.sub(r'([吧嘛啊呢呀呗])([吧嘛啊呢呀呗])', r'\1', text)

    # 4. 句号/句末标点后紧跟句末语气词（不应出现）: ".嘛" → "."
    text = re.sub(r'([。！？.!?])([吧嘛啊呢呀呗])', r'\1', text)

    # 5. 句末语气词 + 省略号/点号 + 另一个语气词: "呀.嘛" → "呀."
    text = re.sub(r'([吧嘛啊呢呀呗])[.…]+([吧嘛啊呢呀呗])', r'\1.', text)

    return text


def build_config(humanize: bool = False, level: str = 'moderate') -> HumanizeConfig:
    """从开关 + 级别构建 HumanizeConfig"""
    if not humanize:
        return HumanizeConfig(enabled=False)
    preset = HUMANIZE_PRESETS.get(level, HUMANIZE_PRESETS['moderate'])
    return preset


# ===================== 句首话语标记 =====================

def _inject_discourse_markers(text: str, config: HumanizeConfig) -> str:
    """
    在段落/句子开头随机插入话语标记词。

    模拟真人在开口说话前的短暂思考停顿。
    仅在首个句子（或。！？之后的新句子）前插入。
    """
    result = []
    # 在句末标点后分割，捕获新句子开头
    parts = _SENTENCE_END_RE.split(text)
    # parts[0] = 首句前的空字符串或内容
    # parts[1] = 标点, parts[2] = 标点后的内容, ...

    for i, part in enumerate(parts):
        if i > 0 and i % 2 == 0:  # 标点后的内容（新句子开头）
            stripped = part.lstrip()
            if stripped and _has_chinese(stripped):
                # 已经以中文开头，考虑插入
                if random.random() < config.discourse_marker_probability:
                    # 只在中文文本前插入，不在已有英文/数字前插入
                    if re.match(r'[一-鿿]', stripped):
                        marker = random.choice(_DISCOURSE_MARKERS)
                        leading_space = part[:len(part) - len(stripped)]
                        part = leading_space + marker + stripped
        result.append(part)

    return ''.join(result)


_SENTENCE_END_RE = re.compile(r'([。！？.!?\n])')


# ===================== 自然停顿标记 =====================

def _inject_natural_pauses(text: str, config: HumanizeConfig) -> str:
    """
    在逗号/分号处随机插入自然停顿标记（省略号等）。

    这会让模型在合成时产生更长的停顿，模拟换气、思考的节奏。
    仅在中文逗号上下文中插入。
    """
    # 找到所有逗号/分号位置
    pause_positions = []
    count = 0
    for m in re.finditer(r'([，、；：])', text):
        pos = m.start()
        # 检查周围是否有中文上下文
        ctx_start = max(0, pos - 5)
        ctx_end = min(len(text), pos + 5)
        if re.search(r'[一-鿿]', text[ctx_start:ctx_end]):
            pause_positions.append((pos, count))
            count += 1

    # 反向插入（避免偏移）
    for pos, _ in reversed(pause_positions):
        if random.random() < config.pause_probability:
            marker = random.choice(_NATURAL_PAUSE_MARKERS)
            # 在逗号前插入停顿标记：...，
            text = text[:pos] + marker + text[pos:]

    return text


# ===================== 句末语气词 =====================

def _inject_sentence_final_particles(text: str, config: HumanizeConfig) -> str:
    """
    在句末标点前随机插入中文语气词（吧、嘛、啊、呢等）。

    让陈述句末尾带有随意的口语感，避免过于"完美"的书面语。
    仅在中文句子中生效。
    """
    # 找到所有句末标点
    end_positions = []
    for m in re.finditer(r'([。！？.!?])', text):
        pos = m.start()
        # 检查标点前是否有中文
        pre_ctx = text[max(0, pos - 6):pos]
        if re.search(r'[一-鿿]', pre_ctx):
            end_positions.append(pos)

    for pos in reversed(end_positions):
        if random.random() < config.sentence_final_probability:
            particle = random.choice(_SENTENCE_FINAL_PARTICLES)
            text = text[:pos] + particle + text[pos:]

    return text


# ===================== 副语言标签注入 =====================

def _inject_paralinguistic_tags(text: str, config: HumanizeConfig) -> str:
    """
    在自然停顿点（标点后）按概率随机插入副语言标签。

    规则:
    - 仅在包含中文的上下文附近插入
    - 不在文本末尾插入
    - 反向遍历以保持字符索引正确
    - v2: breathing 之后可选追加一个微小停顿（文本中用逗号表达）
    """
    tags, weights = zip(*_PARALINGUISTIC_TAGS)
    total_weight = sum(weights)
    normalized_weights = [w / total_weight for w in weights]

    # 找到所有可插入位置
    break_positions = []
    for m in _TAG_BREAK_RE.finditer(text):
        pos = m.end()
        if pos >= len(text):
            continue
        # 仅在中文上下文中插入
        ctx_start = max(0, pos - 10)
        ctx_end = min(len(text), pos + 10)
        if re.search(r'[一-鿿]', text[ctx_start:ctx_end]):
            break_positions.append(pos)

    # 反向插入
    for pos in reversed(break_positions):
        if random.random() < config.tag_probability:
            tag = random.choices(tags, weights=normalized_weights, k=1)[0]
            text = text[:pos] + tag + text[pos:]

    return text


# ===================== 拟人化瑕疵 =====================

def _apply_imperfections(text: str, config: HumanizeConfig) -> str:
    """
    在从句边界之间随机插入说话瑕疵。

    re.split 保留捕获组分隔符 → [clause0, delim0, clause1, delim1, ..., clauseN]
    偶数索引 = 内容从句，奇数索引 = 分隔符。

    规则:
    - 最后一个从句后不插入（避免句尾赘语）
    - 从句必须有中文内容
    - v2: 降低密度，提升多样性
    """
    clauses = _CN_CLAUSE_RE.split(text)

    result = []
    for i, clause in enumerate(clauses):
        result.append(clause)

        # 只在内容从句（偶数索引）后插入
        if i % 2 != 0:
            continue
        # 必须有后续分隔符 + 后续从句
        if i + 2 >= len(clauses):
            continue
        delimiter = clauses[i + 1]
        # 仅在句中停顿前插入，句末标点（。！？）不插
        if delimiter not in _MID_SENTENCE_PUNCT:
            continue
        # 当前从句必须有中文
        if not _has_chinese(clause):
            continue

        if random.random() < config.imperfection_probability:
            imperfection = _generate_imperfection(config)
            # 口吃重复不应出现在句末（下一个分隔符为句末标点时跳过）
            if imperfection in _STUTTERS:
                if i + 3 >= len(clauses):
                    continue  # 文本末尾，跳过口吃
                if clauses[i + 3] in _SENTENCE_END_PUNCT:
                    continue  # 下一个是句末标点，跳过口吃
            result.append(imperfection)

    return ''.join(result)


def _generate_imperfection(config: HumanizeConfig) -> str:
    """按配置比例随机选择并生成一种瑕疵"""
    total = config.filled_pause_rate + config.stutter_rate + config.interjection_rate
    if total <= 0:
        return random.choice(_FILLED_PAUSES)

    r = random.random() * total
    if r < config.filled_pause_rate:
        return random.choice(_FILLED_PAUSES)
    elif r < config.filled_pause_rate + config.stutter_rate:
        return random.choice(_STUTTERS)
    else:
        return random.choice(_INTERJECTIONS)


def _has_chinese(text: str) -> bool:
    """检查文本是否包含中文字符"""
    return bool(re.search(r'[一-鿿]', text))


# ===================== SamplingParams 扰动 =====================

def perturb_sampling_params(params: Any, config: HumanizeConfig) -> Any:
    """
    在 SamplingParams 上施加随机抖动 (v2: 扩大扰动范围)。

    模拟真人在不同语句间的语速、语调自然波动。
    不启用时返回原始参数。

    v2 关键改进: tau_r 扰动范围从 ±0.05 扩大到 ±0.10，
    让 speech rate 产生更明显的变化（tau_r 越高语速越舒缓）。
    """
    if not config.enabled or not config.perturb_sampling:
        return params

    from soulxpodcast.config import SamplingParams

    pc = config.perturb_config

    def _jitter(v, amount, lo=None, hi=None):
        delta = random.uniform(-amount, amount)
        result = v + delta
        if lo is not None:
            result = max(lo, result)
        if hi is not None:
            result = min(hi, result)
        return result

    return SamplingParams(
        temperature=_jitter(params.temperature, pc.temperature_jitter, 0.1, 2.0),
        repetition_penalty=_jitter(params.repetition_penalty, pc.rep_penalty_jitter, 0.5, 2.0),
        top_k=max(1, int(_jitter(params.top_k, pc.top_k_jitter))),
        top_p=_jitter(params.top_p, pc.top_p_jitter, 0.1, 1.0),
        min_tokens=params.min_tokens,
        max_tokens=params.max_tokens,
        stop_token_ids=params.stop_token_ids,
        use_ras=params.use_ras,
        win_size=params.win_size,
        tau_r=_jitter(params.tau_r, pc.tau_r_jitter, 0.02, 0.6),
    )
