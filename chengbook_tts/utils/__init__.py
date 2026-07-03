"""
工具函数模块
-----------
audio:  音频处理（归一化、静音裁剪、WAV I/O）
text:   中文文本预处理（jieba 分词 + [] 包裹）
logging: 统一日志配置
"""

from chengbook_tts.utils.audio import normalize_audio, trim_silence, save_wav, load_wav
from chengbook_tts.utils.text import preprocess_chinese_text
from chengbook_tts.utils.logging import setup_logging

__all__ = [
    'normalize_audio',
    'trim_silence',
    'save_wav',
    'load_wav',
    'preprocess_chinese_text',
    'setup_logging',
]
