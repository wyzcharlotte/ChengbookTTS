"""
音色预设配置
-----------
定义所有预设音色的元信息。
每个音色包含: name (显示名)、wav (提示音频路径)、description (说明)
"""

import os
from pathlib import Path

# prompts 基础路径（预设音色）—— 实际音频通过符号链接指向源项目
PROMPTS_BASE = Path(__file__).resolve().parent.parent / 'prompts' / 'cosyvoice3'

VOICES: dict[str, dict] = {
    'woman': {
        'name': '女客户-小诚',
        'wav': str(PROMPTS_BASE / 'woman.wav'),
        'description': '年轻女客户，适合模拟普通消费者来电',
    },
    'man': {
        'name': '男客户-老书',
        'wav': str(PROMPTS_BASE / 'man.wav'),
        'description': '中年男客户，适合模拟商务/专业场景来电',
    },
}


def get_custom_voice_path(voice_id: str) -> str:
    """获取自定义音色的 WAV 路径"""
    from chengbook_tts.config.settings import settings
    return str(settings.CUSTOM_VOICES_DIR / f'custom_{voice_id}.wav')


def get_voices(include_custom: bool = True) -> dict:
    """
    获取所有可用音色（预设 + 已注册的自定义音色）

    参数:
        include_custom: 是否包含自定义音色
    返回:
        {voice_id: {name, wav, description}}
    """
    voices = dict(VOICES)
    if include_custom:
        # 从 custom_voices.json 加载已注册的自定义音色
        from chengbook_tts.config.settings import settings
        import json
        json_path = settings.CUSTOM_VOICES_JSON
        if json_path.exists():
            try:
                with open(json_path, 'r', encoding='utf-8') as f:
                    custom = json.load(f)
                # 把 wav 相对路径解析为绝对路径
                for voice_id, meta in custom.items():
                    wav = meta.get('wav', '')
                    if wav and not os.path.isabs(wav):
                        meta['wav'] = str(settings.CUSTOM_VOICES_DIR / wav)
                voices.update(custom)
            except Exception:
                pass
    return voices
