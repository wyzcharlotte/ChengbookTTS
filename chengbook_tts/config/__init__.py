"""
配置模块
-------
导出: settings 全局设置, VOICES/EMOTIONS 预设, get_voices()/get_emotions()
"""

from chengbook_tts.config.settings import settings
from chengbook_tts.config.voices import VOICES, get_voices, get_custom_voice_path
from chengbook_tts.config.emotions import EMOTIONS, get_emotions

__all__ = [
    'settings',
    'VOICES',
    'EMOTIONS',
    'get_voices',
    'get_emotions',
    'get_custom_voice_path',
]
