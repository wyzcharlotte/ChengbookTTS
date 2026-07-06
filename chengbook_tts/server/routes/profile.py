"""
Profile API 路由
---------------
GET /api/profile — 读取当前配置
POST /api/profile — 更新配置
"""

import json
import logging

from fastapi import APIRouter, Depends, HTTPException

from chengbook_tts.server.models import ProfileRequest
from chengbook_tts.config.settings import settings

router = APIRouter(tags=['Profile'])


@router.get('/profile')
async def get_profile():
    """读取当前配置文件"""
    profile_path = settings.PROFILES_FILE
    if profile_path.exists():
        try:
            with open(profile_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f'读取配置文件失败: {e}')

    # 返回默认值
    return {
        'voice': settings.DEFAULT_VOICE,
        'emotion': settings.DEFAULT_EMOTION,
        'speed': settings.DEFAULT_SPEED,
        'segment': settings.TEXT_SEGMENT,
        'model_type': settings.MODEL_TYPE,
        'humanize': settings.HUMANIZE_ENABLED,
        'humanizeLevel': settings.HUMANIZE_LEVEL,
    }


@router.post('/profile')
async def update_profile(req: ProfileRequest):
    """更新配置文件"""
    profile_path = settings.PROFILES_FILE

    # 读取现有配置
    current = {}
    if profile_path.exists():
        try:
            with open(profile_path, 'r', encoding='utf-8') as f:
                current = json.load(f)
        except Exception:
            pass

    # 合并更新
    if req.voice is not None:
        current['voice'] = req.voice
    if req.emotion is not None:
        current['emotion'] = req.emotion
    if req.speed is not None:
        current['speed'] = req.speed
    if req.segment is not None:
        current['segment'] = req.segment
    if req.humanize is not None:
        current['humanize'] = req.humanize
    if req.humanize_level is not None:
        current['humanizeLevel'] = req.humanize_level

    # 保存
    profile_path.parent.mkdir(parents=True, exist_ok=True)
    with open(profile_path, 'w', encoding='utf-8') as f:
        json.dump(current, f, ensure_ascii=False, indent=2)

    logging.info(f'Profile updated: {current}')
    return {'success': True, 'profile': current}
