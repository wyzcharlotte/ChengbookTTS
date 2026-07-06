"""
SoulXPodcast 拟人化文本预处理器
-------------------------------
通过注入副语言标签 (<|laughter|>, <|sigh|> 等) 和生成说话瑕疵
(填充词、重复、纠错) 让合成语音更贴近真人说话习惯。

仅在 SoulXPodcast 引擎下有效 — Qwen3 tokenizer 能识别这些特殊 token。
"""

import re
import random
from dataclasses import dataclass, field
from typing import Any, List, Tuple


# ===================== 副语言标签定义 =====================

# (标签, 权重) — 权重越高出现频率越高
_PARALINGUISTIC_TAGS: List[Tuple[str, int]] = [
    ('<|breathing|>',        8),   # 呼吸 — 最自然最常见
    ('<|sigh|>',             3),   # 叹气 — 中等频率
    ('<|coughing|>',         2),   # 咳嗽 — 偶发
    ('<|throat_clearing|>',  2),   # 清嗓子 — 偶发
]

# 自然停顿点 — 在这些标点后可以插入副语言标签
_TAG_BREAK_RE = re.compile(r'([。！？，、；：\n!?,:;])')

# 中文从句边界 — 用于分割句子以插入瑕疵
_CN_CLAUSE_RE = re.compile(r'([。！？，、；：\n]|(?<=[a-zA-Z])[.?!,;:])')

# 句中停顿（可插入瑕疵的位置）— 句末标点。！？不可插入
_MID_SENTENCE_PUNCT = {'，', '、', '；', '：'}


# ===================== 拟人化瑕疵模板 =====================

# 填充词
_FILLED_PAUSES: List[str] = [
    '嗯，',
    '啊，',
    '那个，',
    '就是说，',
    '然后呢，',
]

# 口吃/重复
_STUTTERS: List[str] = [
    '这个这个',
    '就就',
    '然后然后',
    '那那',
    '这这',
]

# 纠错标记 → 改为轻度语气词/犹豫，避免"啊说错了"这种太突兀的表达
_INTERJECTIONS: List[str] = [
    '呃，',
    '哎，',
    '哦，',
    '呃等一下，',
    '不对，应该是呃，',
]


# ===================== SamplingParams 扰动配置 =====================

@dataclass
class HumanizePerturbConfig:
    """SamplingParams 随机扰动范围"""
    temperature_jitter: float = 0.05
    top_k_jitter: int = 10
    top_p_jitter: float = 0.05
    rep_penalty_jitter: float = 0.05
    tau_r_jitter: float = 0.05


# ===================== 主配置 =====================

@dataclass
class HumanizeConfig:
    """拟人化预处理总配置"""
    enabled: bool = False

    # 副语言标签注入
    inject_tags: bool = True
    tag_probability: float = 0.15      # 每个停顿点插入标签的概率

    # 拟人化瑕疵
    add_imperfections: bool = True
    imperfection_probability: float = 0.08  # 每个从句边界插入瑕疵的概率

    # 瑕疵类型比例 (和为 1.0)
    filled_pause_rate: float = 0.65
    stutter_rate: float = 0.20
    interjection_rate: float = 0.15

    # 采样参数扰动
    perturb_sampling: bool = True
    perturb_config: HumanizePerturbConfig = field(default_factory=HumanizePerturbConfig)


# 预设强度
HUMANIZE_PRESETS = {
    'default':  HumanizeConfig(enabled=False),
    'light':    HumanizeConfig(enabled=True, tag_probability=0.08, imperfection_probability=0.03),
    'moderate': HumanizeConfig(enabled=True, tag_probability=0.15, imperfection_probability=0.08),
    'heavy':    HumanizeConfig(enabled=True, tag_probability=0.25, imperfection_probability=0.15),
}


# ===================== 主入口 =====================

def apply_humanization(text: str, config: HumanizeConfig) -> str:
    """
    对输入文本施加拟人化预处理。

    返回预处理后的文本。如果 config.enabled 为 False 则原样返回。
    每次调用使用不同的随机种子，确保同一输入产生不同结果。
    """
    if not config.enabled:
        return text

    if not text or not text.strip():
        return text

    random.seed()  # 每次调用重新播种，保证随机性

    if config.inject_tags:
        text = _inject_paralinguistic_tags(text, config)

    if config.add_imperfections:
        text = _apply_imperfections(text, config)

    return text


def build_config(humanize: bool = False, level: str = 'moderate') -> HumanizeConfig:
    """从开关 + 级别构建 HumanizeConfig"""
    if not humanize:
        return HumanizeConfig(enabled=False)
    preset = HUMANIZE_PRESETS.get(level, HUMANIZE_PRESETS['moderate'])
    return preset


# ===================== 副语言标签注入 =====================

def _inject_paralinguistic_tags(text: str, config: HumanizeConfig) -> str:
    """
    在自然停顿点（标点后）按概率随机插入副语言标签。

    规则:
    - 仅在包含中文的上下文附近插入
    - 不在文本末尾插入
    - 反向遍历以保持字符索引正确
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
    在 SamplingParams 上施加随机抖动。

    模拟真人在不同语句间的语速、语调自然波动。
    不启用时返回原始参数。
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
        tau_r=_jitter(params.tau_r, pc.tau_r_jitter, 0.01, 0.5),
    )
