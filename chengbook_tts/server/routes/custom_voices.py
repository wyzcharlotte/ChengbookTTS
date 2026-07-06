"""
自定义音色 API 路由
------------------
POST /api/voices/custom — 上传自定义音色
DELETE /api/voices/custom/{voice_id} — 删除自定义音色
"""

import os
import json
import uuid
import time
import logging
import asyncio
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form

from chengbook_tts.engines.base import TTSEngine
from chengbook_tts.server.dependencies import get_engine
from chengbook_tts.server import concurrency
from chengbook_tts.config.settings import settings

router = APIRouter(tags=['Custom Voices'])


# ---------- 持久化辅助 ----------

def _load_custom_voices_json() -> dict:
    json_path = settings.CUSTOM_VOICES_JSON
    if not json_path.exists():
        return {}
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        return {}


def _save_custom_voices_json(data: dict):
    settings.CUSTOM_VOICES_DIR.mkdir(parents=True, exist_ok=True)
    with open(settings.CUSTOM_VOICES_JSON, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def _add_voice_to_json(voice_id: str, metadata: dict):
    data = _load_custom_voices_json()
    data[voice_id] = metadata
    _save_custom_voices_json(data)


def _remove_voice_from_json(voice_id: str):
    data = _load_custom_voices_json()
    if voice_id in data:
        del data[voice_id]
        _save_custom_voices_json(data)


def load_custom_voices_on_startup():
    """服务启动时恢复自定义音色"""
    engine = get_engine()
    data = _load_custom_voices_json()
    if not data:
        logging.info('No custom voices to restore')
        return

    restored = 0
    for voice_id, meta in data.items():
        wav_path = settings.CUSTOM_VOICES_DIR / meta.get('wav', '')
        if not wav_path.exists():
            logging.warning(f'Custom voice [{voice_id}] wav not found: {wav_path}')
            continue
        if engine.register_voice(voice_id, str(wav_path), meta.get('name', voice_id),
                                  meta.get('description', ''), meta.get('prompt_text', '')):
            restored += 1
        else:
            logging.warning(f'Custom voice [{voice_id}] restore failed')

    if restored > 0:
        logging.info(f'Restored {restored} custom voice(s)')


# ---------- 路由 ----------

@router.post('/voices/custom')
async def create_custom_voice(
    file: UploadFile = File(..., description='音频文件 (WAV)'),
    name: str = Form(..., description='音色名称'),
    description: str = Form('', description='音色描述（可选）'),
    prompt_text: str = Form('', description='参考音频对应的转录文本（可选，上传后可提升音色克隆质量）'),
    engine: TTSEngine = Depends(get_engine),
):
    """上传自定义音色"""
    # 校验数量上限
    custom_count = sum(1 for vid in engine.voice_ids if engine.is_custom_voice(vid))
    if custom_count >= settings.MAX_CUSTOM_VOICES:
        raise HTTPException(
            status_code=400,
            detail=f'自定义音色已达上限 ({settings.MAX_CUSTOM_VOICES})，请先删除旧音色',
        )

    # 校验格式
    original_filename = file.filename or 'upload.wav'
    if not original_filename.lower().endswith('.wav'):
        raise HTTPException(status_code=400, detail='仅支持 WAV 格式')

    # 校验大小
    content = await file.read()
    if len(content) > settings.MAX_UPLOAD_FILE_SIZE:
        raise HTTPException(
            status_code=400,
            detail=f'文件过大 ({len(content)} bytes)，上限 {settings.MAX_UPLOAD_FILE_SIZE // (1024*1024)} MB',
        )

    # 生成 voice_id
    voice_id = f'custom_{uuid.uuid4().hex[:8]}'

    # 保存 WAV
    settings.CUSTOM_VOICES_DIR.mkdir(parents=True, exist_ok=True)
    wav_filename = f'{voice_id}.wav'
    wav_path = settings.CUSTOM_VOICES_DIR / wav_filename
    with open(wav_path, 'wb') as f:
        f.write(content)

    # 在引擎中注册
    loop = asyncio.get_running_loop()
    try:
        success = await loop.run_in_executor(
            concurrency.get_executor(),
            engine.register_voice, voice_id, str(wav_path), name, description or '', prompt_text or '',
        )
    except NotImplementedError:
        try:
            os.unlink(wav_path)
        except OSError:
            pass
        raise HTTPException(
            status_code=400,
            detail=f'当前引擎 ({engine.engine_name}) 不支持自定义音色注册',
        )
    except Exception as e:
        # 清理残留文件
        try:
            os.unlink(wav_path)
        except OSError:
            pass
        logging.error(f'Voice registration failed: {e}', exc_info=True)
        raise HTTPException(status_code=500, detail=f'音色注册失败: {str(e)}')

    # 持久化
    _add_voice_to_json(voice_id, {
        'name': name,
        'wav': wav_filename,
        'description': description or f'用户自定义音色: {name}',
        'prompt_text': prompt_text or '',
        'created_at': time.strftime('%Y-%m-%dT%H:%M:%S'),
    })

    logging.info(f'Custom voice created: {voice_id} "{name}"')
    return {
        'success': True,
        'voice_id': voice_id,
        'name': name,
        'message': f'自定义音色 "{name}" 已创建，合成时使用 voice="{voice_id}"',
    }


@router.delete('/voices/custom/{voice_id}')
async def delete_custom_voice(voice_id: str, engine: TTSEngine = Depends(get_engine)):
    """删除自定义音色（前 4 个音色受保护）"""
    if voice_id not in engine.voice_ids:
        raise HTTPException(status_code=404, detail=f'音色不存在: {voice_id}')

    # 前 2 个音色（预设）受保护，其余均可删除
    protected = set(engine.voice_ids[:2])
    if voice_id in protected:
        raise HTTPException(
            status_code=400,
            detail=f'"{voice_id}" 是预设音色，不可删除',
        )

    info = engine.voice_info(voice_id)
    voice_name = info.get('name', voice_id)

    # 清理 WAV 文件
    wav_path = settings.CUSTOM_VOICES_DIR / f'{voice_id}.wav'
    try:
        if wav_path.exists():
            os.unlink(wav_path)
    except OSError as e:
        logging.warning(f'Failed to remove wav: {wav_path}: {e}')

    # 引擎注销
    engine.unregister_voice(voice_id)

    # 清理 JSON
    _remove_voice_from_json(voice_id)

    logging.info(f'Custom voice deleted: {voice_id} "{voice_name}"')
    return {
        'success': True,
        'voice_id': voice_id,
        'name': voice_name,
        'message': f'自定义音色 "{voice_name}" 已删除',
    }
