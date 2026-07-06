"""
全局配置中心
-----------
所有设置通过环境变量 + 默认值统一管理。
模型专属配置见 model_configs/*.yaml。
"""

import os
from pathlib import Path
from dataclasses import dataclass, field


@dataclass
class Settings:
    """ChengbookTTS 全局设置 (环境变量可覆盖)"""

    # ====== 项目路径 ======
    ROOT_DIR: Path = Path(__file__).resolve().parent.parent
    VENDOR_DIR: Path = ROOT_DIR / 'vendor'
    PROMPTS_DIR: Path = ROOT_DIR / 'prompts'
    PROFILES_DIR: Path = ROOT_DIR / 'profiles'
    PROFILES_FILE: Path = PROFILES_DIR / 'current.json'
    CUSTOM_VOICES_DIR: Path = PROMPTS_DIR / 'custom'
    CUSTOM_VOICES_JSON: Path = ROOT_DIR / 'custom_voices.json'

    # ====== 模型选择 ======
    # 'auto' 自动检测 | 'cosyvoice3' | 'soulxpodcast' | 'indextts_v1' | 'indextts_v2'
    MODEL_TYPE: str = os.environ.get('MODEL_TYPE', 'cosyvoice3')

    # ====== 服务配置 ======
    HOST: str = os.environ.get('HOST', '0.0.0.0')
    PORT: int = int(os.environ.get('PORT', '8080'))
    API_KEY: str = os.environ.get('API_KEY', '')

    # ====== 加速选项 ======
    FP16: bool = os.environ.get('FP16', '1').lower() in ('1', 'true', 'yes')
    LOAD_TRT: bool = os.environ.get('LOAD_TRT', '0').lower() in ('1', 'true', 'yes')

    # ====== 并发控制 ======
    MAX_CONCURRENT: int = int(os.environ.get('MAX_CONCURRENT', '1'))

    # ====== 文本限制 ======
    MAX_TEXT_LENGTH: int = 2000

    # ====== 自定义音色 ======
    MAX_CUSTOM_VOICES: int = int(os.environ.get('MAX_CUSTOM_VOICES', '20'))
    MAX_UPLOAD_FILE_SIZE: int = 10 * 1024 * 1024  # 10 MB

    # ====== 默认值 ======
    DEFAULT_VOICE: str = 'woman'
    DEFAULT_EMOTION: str = 'calm'
    DEFAULT_SPEED: float = 1.0
    SPEED_MIN: float = 0.5
    SPEED_MAX: float = 2.0

    # ====== 文本预处理 ======
    TEXT_PREPROCESS: bool = os.environ.get('TEXT_PREPROCESS', '1').lower() in ('1', 'true', 'yes')
    TEXT_SEGMENT: bool = os.environ.get('TEXT_SEGMENT', '1').lower() in ('1', 'true', 'yes')

    # ====== 拟人化预处理（仅 SoulXPodcast） ======
    HUMANIZE_ENABLED: bool = False
    HUMANIZE_LEVEL: str = 'moderate'

    # ====== 合成超时 ======
    TTS_TIMEOUT: int = int(os.environ.get('TTS_TIMEOUT', '300'))


# 全局单例
settings = Settings()
