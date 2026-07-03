"""
工具函数 — 中文文本预处理
--------------------------
jieba 分词 + 歧义词 [] 包裹，减少 LLM 断句错误。
从原 engine.py 迁移而来。
"""

import re
import logging

try:
    import jieba
    _SEG_AVAILABLE = True
except ImportError:
    _SEG_AVAILABLE = False

# 匹配中文字符
_RE_CHINESE_CHAR = re.compile(r'[一-鿿㐀-䶿]')


def _is_chinese_2char(s: str) -> bool:
    """是否为纯中文二字词"""
    return len(s) == 2 and _RE_CHINESE_CHAR.search(s) is not None


def preprocess_chinese_text(text: str, segment: bool = False) -> str:
    """
    中文文本预处理: 相邻二字词自动合并为词组，再用 [词] 包裹。

    合并规则:
        jieba 输出的连续二字词拼成一个整体（如 "发票"+"抬头"→"发票抬头"），
        单字和五字以上长词保持原样不加括号。

    例如:
        "我的发票抬头是个人，可以开个人发票吗"
        → "我的 [发票抬头] 是 [个人] ， [可以] 开 [个人发票] 吗"

    参数:
        text: 原始文本
        segment: False 时返回原文

    返回:
        预处理后的文本
    """
    if not _SEG_AVAILABLE:
        return text
    if not _RE_CHINESE_CHAR.search(text):
        return text
    if not segment:
        return text

    words = jieba.lcut(text)

    # 第一步: 合并相邻的二字中文词
    merged = []
    buf = []
    for w in words:
        stripped = w.strip()
        if not stripped:
            continue
        if _is_chinese_2char(stripped):
            buf.append(stripped)
        else:
            if buf:
                merged.append(''.join(buf))
                buf.clear()
            merged.append(stripped)
    if buf:
        merged.append(''.join(buf))

    # 第二步: 合并后的结果加括号 — 原二字词及合并词组都加
    result_parts = []
    for w in merged:
        if _RE_CHINESE_CHAR.search(w) is not None and 2 <= len(w) <= 4:
            result_parts.append(f' [{w}] ')
        else:
            result_parts.append(w)

    result = ''.join(result_parts)
    result = re.sub(r' {2,}', ' ', result)
    return result.strip()
