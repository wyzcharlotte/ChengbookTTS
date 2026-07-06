"""
Pydantic 请求/响应模型
---------------------
定义所有 API 的请求体和响应体。
"""

from typing import Optional, List
from pydantic import BaseModel, Field

from chengbook_tts.config.settings import settings


# ---------- 原生 API 模型 ----------

class TTSRequest(BaseModel):
    text: str = Field(..., description='待合成文本', max_length=settings.MAX_TEXT_LENGTH)
    voice: str = Field(settings.DEFAULT_VOICE, description='音色 ID')
    emotion: str = Field(settings.DEFAULT_EMOTION, description='情绪 ID')
    speed: float = Field(settings.DEFAULT_SPEED, description=f'语速 ({settings.SPEED_MIN}~{settings.SPEED_MAX})',
                         ge=settings.SPEED_MIN, le=settings.SPEED_MAX)
    segment: bool = Field(False, description='是否启用分词+[]预处理（默认关闭，与原始 CosyVoice3 行为一致）')
    humanize: bool = Field(False, description='是否启用拟人化预处理（仅 SoulXPodcast）')
    humanize_level: str = Field('moderate', description='拟人化强度: light / moderate / heavy')


class TTSStreamRequest(BaseModel):
    text: str = Field(..., description='待合成文本', max_length=settings.MAX_TEXT_LENGTH)
    voice: str = Field(settings.DEFAULT_VOICE, description='音色 ID')
    emotion: str = Field(settings.DEFAULT_EMOTION, description='情绪 ID')
    segment: bool = Field(False, description='是否启用分词+[]预处理（默认关闭，与原始 CosyVoice3 行为一致）')


# ---------- OpenAI 兼容 API 模型 ----------

class OpenAISpeechRequest(BaseModel):
    """OpenAI /v1/audio/speech 兼容请求体"""
    model: str = Field('tts-1', description='模型名称')
    input: str = Field(..., description='待合成文本', max_length=settings.MAX_TEXT_LENGTH)
    voice: str = Field('alloy', description='音色 (alloy/echo/fable/nova/onyx/shimmer 或 woman/man)')
    response_format: str = Field('wav', description='音频格式 (mp3/opus/aac/flac/wav/pcm)')
    speed: float = Field(1.0, description='语速 0.25~4.0', ge=0.25, le=4.0)


class OpenAIModelItem(BaseModel):
    id: str
    object: str = 'model'
    created: int = 0
    owned_by: str = 'chengshuji'


class OpenAIModelList(BaseModel):
    object: str = 'list'
    data: List[OpenAIModelItem]


# ---------- Profile 模型 ----------

class ProfileRequest(BaseModel):
    voice: Optional[str] = None
    emotion: Optional[str] = None
    speed: Optional[float] = Field(None, ge=settings.SPEED_MIN, le=settings.SPEED_MAX)
    segment: Optional[bool] = None
    humanize: Optional[bool] = None
    humanize_level: Optional[str] = None
