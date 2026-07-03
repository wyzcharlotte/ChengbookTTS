"""
OpenAI 兼容 API 路由
-------------------
/v1/audio/speech, /v1/models, /v1/health
供诚太乙等外部产品调用
"""

import io
import time
import logging
import asyncio

import numpy as np
import soundfile as sf
from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import Response

from chengbook_tts.engines.base import TTSEngine
from chengbook_tts.server.dependencies import get_engine, verify_api_key
from chengbook_tts.server.models import OpenAISpeechRequest, OpenAIModelItem, OpenAIModelList
from chengbook_tts.server.audio_utils import audio_to_bytes, FORMAT_MIME
from chengbook_tts.server import concurrency
from chengbook_tts.server.history import get_history_manager
from chengbook_tts.config.settings import settings

router = APIRouter(tags=['OpenAI Compatible API'])

# OpenAI voice → Chengbook voice 映射
OPENAI_VOICE_MAP = {
    'alloy': 'woman',
    'nova': 'woman',
    'shimmer': 'woman',
    'echo': 'man',
    'fable': 'man',
    'onyx': 'man',
    'woman': 'woman',
    'man': 'man',
}
OPENAI_DEFAULT_EMOTION = 'calm'


def _get_profile_voice_emotion():
    """从 profiles/current.json 读取当前偏好"""
    profile_path = settings.PROFILES_FILE
    voice = settings.DEFAULT_VOICE
    emotion = OPENAI_DEFAULT_EMOTION
    if profile_path.exists():
        try:
            import json
            with open(profile_path, 'r', encoding='utf-8') as f:
                profile = json.load(f)
            voice = profile.get('voice', voice)
            emotion = profile.get('emotion', emotion)
        except Exception:
            pass
    return voice, emotion


# ---------- /v1/models ----------

@router.get('/models')
async def list_models(_api_key: str = Depends(verify_api_key)):
    items = [
        OpenAIModelItem(id='tts-1'),
        OpenAIModelItem(id='tts-1-hd'),
        OpenAIModelItem(id='chengshuji-tts-v1'),
    ]
    return OpenAIModelList(data=items)


# ---------- /v1/health ----------

@router.get('/health')
async def openai_health(_api_key: str = Depends(verify_api_key)):
    return {'status': 'ok'}


# ---------- /v1/audio/speech ----------

@router.post('/audio/speech')
@router.get('/audio/speech')  # 兼容 GET
async def audio_speech(
    req: OpenAISpeechRequest = None,
    engine: TTSEngine = Depends(get_engine),
    _api_key: str = Depends(verify_api_key),
):
    """
    OpenAI 兼容 TTS 端点。

    如果通过 GET 请求（诚太乙用 GET），从 query params 解析。
    """
    # GET 请求处理
    if req is None:
        return await _audio_speech_get(engine)

    return await _audio_speech_post(req, engine)


async def _audio_speech_post(req: OpenAISpeechRequest, engine: TTSEngine):
    text = req.input
    fmt = req.response_format or 'wav'
    if fmt not in FORMAT_MIME:
        raise HTTPException(status_code=400, detail=f'不支持的格式: {fmt}')

    # 映射 voice
    internal_voice = OPENAI_VOICE_MAP.get(req.voice)
    if not internal_voice:
        raise HTTPException(status_code=400, detail=f'不支持的音色: {req.voice}')

    # 检查并 fallback
    if internal_voice not in engine.voice_ids:
        profile_voice, _ = _get_profile_voice_emotion()
        internal_voice = profile_voice
        if internal_voice not in engine.voice_ids:
            internal_voice = engine.voice_ids[0]

    _, emotion = _get_profile_voice_emotion()
    if emotion not in engine.emotion_ids:
        emotion = engine.emotion_ids[0]

    t0 = time.time()
    logging.info(
        f'OpenAI TTS: voice={req.voice}→{internal_voice}, fmt={fmt}, '
        f'text_len={len(text)}, text={text[:50]}...'
    )

    try:
        audio = await _run_synthesize_openai(engine, text, internal_voice, emotion, req.speed)
    except Exception as e:
        logging.error(f'OpenAI synthesis failed: {e}', exc_info=True)
        get_history_manager().record(
            text=text,
            voice_name=engine.voice_info(internal_voice).get('name', internal_voice),
            emotion_name=engine.emotion_info(emotion).get('name', emotion),
            speed=req.speed,
            model=f'{engine.engine_name} v{engine.engine_version}',
            streaming=False,
            success=False,
            error=str(e),
        )
        raise HTTPException(status_code=500, detail=f'合成失败: {str(e)}')

    elapsed = time.time() - t0
    duration = len(audio) / engine.sample_rate if engine.sample_rate > 0 else 0
    rtf = elapsed / duration if duration > 0 else 0
    logging.info(
        f'OpenAI TTS done: duration={duration:.2f}s, elapsed={elapsed:.2f}s'
    )

    # 记录合成历史
    get_history_manager().record(
        text=text,
        voice_name=engine.voice_info(internal_voice).get('name', internal_voice),
        emotion_name=engine.emotion_info(emotion).get('name', emotion),
        speed=req.speed,
        model=f'{engine.engine_name} v{engine.engine_version}',
        duration=duration,
        elapsed=elapsed,
        rtf=rtf,
        streaming=False,
        success=True,
    )

    audio_bytes = audio_to_bytes(audio, engine.sample_rate, fmt)

    return Response(
        content=audio_bytes,
        media_type=FORMAT_MIME.get(fmt, 'audio/wav'),
        headers={
            'X-Audio-Duration': f'{duration:.2f}',
            'X-Elapsed': f'{elapsed:.2f}',
            'X-Voice': internal_voice,
            'X-Emotion': emotion,
        },
    )


async def _audio_speech_get(engine: TTSEngine):
    """处理 GET /v1/audio/speech（诚太乙可能用 GET）"""
    raise HTTPException(status_code=400, detail='请使用 POST 请求，提供 JSON Body')


async def _run_synthesize_openai(engine: TTSEngine, text: str, voice: str,
                                  emotion: str, speed: float) -> np.ndarray:
    """OpenAI 路径的合成（不带 segment 参数）"""
    concurrency.inc_pending()
    try:
        async with concurrency.get_semaphore():
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(
                concurrency.get_executor(),
                engine.synthesize, text, voice, emotion, speed, False,
            )
    finally:
        concurrency.dec_pending()
