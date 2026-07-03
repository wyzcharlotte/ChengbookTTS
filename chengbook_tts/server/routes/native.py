"""
原生 API 路由
------------
/api/health, /api/voices, /api/emotions, /api/tts, /api/tts/stream
"""

import io
import time
import queue as sync_queue
import logging
import asyncio

import numpy as np
import soundfile as sf
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import Response, StreamingResponse

from chengbook_tts.engines.base import TTSEngine
from chengbook_tts.server.dependencies import get_engine
from chengbook_tts.server.models import TTSRequest, TTSStreamRequest
from chengbook_tts.server import concurrency
from chengbook_tts.server.history import get_history_manager

router = APIRouter(tags=['Native API'])


def _check_voice(engine: TTSEngine, voice: str):
    if voice not in engine.voice_ids:
        raise HTTPException(
            status_code=400,
            detail=f'不支持的音色: {voice}，可用: {engine.voice_ids}'
        )


def _check_emotion(engine: TTSEngine, emotion: str):
    if emotion not in engine.emotion_ids:
        raise HTTPException(
            status_code=400,
            detail=f'不支持的情绪: {emotion}，可用: {engine.emotion_ids}'
        )


# ---------- 健康检查 ----------

@router.get('/health')
async def health(engine: TTSEngine = Depends(get_engine)):
    info = engine.health_info()
    custom_count = sum(1 for vid in engine.voice_ids if engine.is_custom_voice(vid))
    return {
        'status': 'ok',
        'service': 'ChengbookTTS — 统一多模型 TTS 平台',
        **info,
        'capabilities': {
            'streaming': engine.supports_streaming,
            'emotion': engine.supports_emotion,
            'multi_speaker': engine.supports_multi_speaker,
            'speed_control': engine.supports_speed_control,
            'segmentation': engine.supports_segmentation,
        },
        'voices': {vid: engine.voice_info(vid).get('name', vid) for vid in engine.voice_ids},
        'emotions': {eid: engine.emotion_info(eid).get('name', eid) for eid in engine.emotion_ids},
        'custom_voices': custom_count,
        'concurrency': {
            'max_concurrent': concurrency.get_semaphore()._value if concurrency.get_semaphore() else 0,
            'pending': concurrency.get_pending(),
            'total_completed': concurrency.get_total(),
        },
    }


# ---------- 音色/情绪列表 ----------

@router.get('/voices')
async def list_voices(engine: TTSEngine = Depends(get_engine)):
    from chengbook_tts.config.settings import settings
    voices_list = []
    for vid in engine.voice_ids:
        info = engine.voice_info(vid)
        voices_list.append({
            'id': vid,
            'name': info.get('name', vid),
            'description': info.get('description', ''),
            'is_custom': engine.is_custom_voice(vid),
        })
    return {
        'voices': voices_list,
        'default': settings.DEFAULT_VOICE,
        'custom_voices_count': sum(1 for v in voices_list if v['is_custom']),
    }


@router.get('/emotions')
async def list_emotions(engine: TTSEngine = Depends(get_engine)):
    from chengbook_tts.config.settings import settings
    return {
        'emotions': [
            {
                'id': eid,
                'name': engine.emotion_info(eid).get('name', eid),
                'description': engine.emotion_info(eid).get('description', ''),
            }
            for eid in engine.emotion_ids
        ],
        'default': settings.DEFAULT_EMOTION,
    }


# ---------- 非流式合成 ----------

@router.post('/tts')
async def tts(req: TTSRequest, engine: TTSEngine = Depends(get_engine)):
    _check_voice(engine, req.voice)
    _check_emotion(engine, req.emotion)

    t0 = time.time()
    logging.info(
        f'TTS: voice={req.voice}, emotion={req.emotion}, speed={req.speed}, '
        f'text_len={len(req.text)}, text={req.text[:50]}..., '
        f'queue={concurrency.get_pending()}'
    )

    try:
        audio = await _run_synthesize(engine, req.text, req.voice, req.emotion,
                                      req.speed, segment=req.segment)
    except Exception as e:
        logging.error(f'Synthesis failed: {e}', exc_info=True)
        get_history_manager().record(
            text=req.text,
            voice_name=engine.voice_info(req.voice).get('name', req.voice),
            emotion_name=engine.emotion_info(req.emotion).get('name', req.emotion),
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

    # print() → 直接 stdout，绕开 logger 配置问题
    print(f'[TTS] sr={engine.sample_rate}, audio_shape={audio.shape}, dtype={audio.dtype}, '
          f'amp=[{audio.min():.4f}, {audio.max():.4f}], '
          f'duration={duration:.2f}s, elapsed={elapsed:.2f}s, RTF={rtf:.3f}', flush=True)
    logging.info(f'TTS done: duration={duration:.2f}s, elapsed={elapsed:.2f}s, RTF={rtf:.3f}')

    # 记录合成历史
    get_history_manager().record(
        text=req.text,
        voice_name=engine.voice_info(req.voice).get('name', req.voice),
        emotion_name=engine.emotion_info(req.emotion).get('name', req.emotion),
        speed=req.speed,
        model=f'{engine.engine_name} v{engine.engine_version}',
        duration=duration,
        elapsed=elapsed,
        rtf=rtf,
        streaming=False,
        success=True,
    )

    # 编码为 WAV（显式 PCM_16，确保浏览器兼容）
    buf = io.BytesIO()
    sf.write(buf, audio, engine.sample_rate, format='WAV', subtype='PCM_16')
    buf.seek(0)
    wav_bytes = buf.read()
    print(f'[TTS] WAV: {len(wav_bytes)} bytes, first 4B={wav_bytes[:4]}', flush=True)

    return Response(
        content=wav_bytes,
        media_type='audio/wav',
        headers={
            'X-Audio-Duration': f'{duration:.2f}',
            'X-Elapsed': f'{elapsed:.2f}',
            'X-RTF': f'{rtf:.3f}',
            'X-Voice': req.voice,
            'X-Emotion': req.emotion,
            'X-Speed': str(req.speed),
        },
    )


async def _run_synthesize(engine: TTSEngine, text: str, voice: str, emotion: str,
                          speed: float, segment: bool = True) -> np.ndarray:
    """线程池中执行合成，asyncio 不阻塞"""
    concurrency.inc_pending()
    try:
        async with concurrency.get_semaphore():
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(
                concurrency.get_executor(),
                engine.synthesize, text, voice, emotion, speed, segment,
            )
    finally:
        concurrency.dec_pending()


# ---------- 流式合成 ----------

@router.post('/tts/stream')
async def tts_stream(req: TTSStreamRequest, engine: TTSEngine = Depends(get_engine)):
    _check_voice(engine, req.voice)
    _check_emotion(engine, req.emotion)

    logging.info(
        f'Stream TTS: voice={req.voice}, emotion={req.emotion}, '
        f'text={req.text[:40]}..., queue={concurrency.get_pending()}'
    )

    async def generate():
        t0 = time.time()
        t_first = None
        chunk_count = 0
        total_samples = 0
        stream_error = None

        try:
            async for audio in _stream_generator(
                engine, req.text, req.voice, req.emotion, segment=req.segment
            ):
                samples = len(audio)
                total_samples += samples
                chunk_count += 1

                if t_first is None:
                    t_first = time.time()
                    logging.info(
                        f'  First chunk: {samples}samples '
                        f'({samples / engine.sample_rate:.2f}s), '
                        f'latency={t_first - t0:.3f}s'
                    )

                pcm = (audio * 32767).astype(np.int16).tobytes()
                yield pcm
        except Exception as e:
            stream_error = str(e)
            raise

        finally:
            elapsed = time.time() - t0
            duration = total_samples / engine.sample_rate if engine.sample_rate > 0 else 0
            rtf = elapsed / duration if duration > 0 else 0
            logging.info(
                f'Stream done: {chunk_count} chunks, duration={duration:.2f}s, '
                f'elapsed={elapsed:.2f}s'
            )
            # 记录合成历史
            get_history_manager().record(
                text=req.text,
                voice_name=engine.voice_info(req.voice).get('name', req.voice),
                emotion_name=engine.emotion_info(req.emotion).get('name', req.emotion),
                speed=1.0,
                model=f'{engine.engine_name} v{engine.engine_version}',
                duration=duration,
                elapsed=elapsed,
                rtf=rtf,
                streaming=True,
                success=stream_error is None,
                error=stream_error or '',
            )

    return StreamingResponse(
        generate(),
        media_type='audio/raw',
        headers={
            'X-Audio-SampleRate': str(engine.sample_rate),
            'X-Audio-Format': 'pcm_s16le',
            'X-Audio-Channels': '1',
            'X-Voice': req.voice,
            'X-Emotion': req.emotion,
        },
    )


async def _stream_generator(engine: TTSEngine, text: str, voice: str, emotion: str,
                            segment: bool = True):
    """流式合成 async generator 包装"""
    concurrency.inc_pending()
    q = sync_queue.Queue(maxsize=8)
    loop = asyncio.get_running_loop()

    def _run():
        try:
            for chunk in engine.synthesize_stream(text, voice, emotion, segment=segment):
                q.put(('chunk', chunk))
            q.put(('done', None))
        except Exception as e:
            q.put(('error', e))

    async with concurrency.get_semaphore():
        task = loop.run_in_executor(concurrency.get_executor(), _run)
        try:
            while True:
                status, data = await loop.run_in_executor(None, q.get)
                if status == 'done':
                    break
                elif status == 'error':
                    raise data
                else:
                    yield data
        finally:
            concurrency.dec_pending()
            try:
                await task
            except Exception:
                pass
