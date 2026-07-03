"""
FastAPI 依赖注入
---------------
提供 Depends() 函数，供路由使用。
"""

import logging
from typing import Optional

from fastapi import Depends, HTTPException
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

from chengbook_tts.engines.base import TTSEngine
from chengbook_tts.engines.manager import ModelManager
from chengbook_tts.config.settings import settings

security = HTTPBearer(auto_error=False)


def get_engine() -> TTSEngine:
    """获取当前活跃的 TTS 引擎"""
    manager = ModelManager.get_instance()
    return manager.engine


def get_model_manager() -> ModelManager:
    """获取模型管理器"""
    return ModelManager.get_instance()


async def verify_api_key(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
) -> Optional[str]:
    """
    验证 Bearer Token。
    - 未配置 API_KEY 时跳过认证
    - 配置了则校验 Bearer Token
    """
    if not settings.API_KEY:
        return None
    if credentials is None:
        raise HTTPException(
            status_code=401,
            detail='缺少 API Key，请在 Authorization 头中提供 Bearer Token'
        )
    if credentials.credentials != settings.API_KEY:
        raise HTTPException(status_code=401, detail='API Key 无效')
    return credentials.credentials
