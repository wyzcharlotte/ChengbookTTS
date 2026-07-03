"""
路由模块
-------
按功能拆分的 FastAPI 路由。
"""

from chengbook_tts.server.routes.native import router as native_router
from chengbook_tts.server.routes.custom_voices import router as custom_voices_router
from chengbook_tts.server.routes.openai import router as openai_router
from chengbook_tts.server.routes.profile import router as profile_router
from chengbook_tts.server.routes.model import router as model_router

__all__ = [
    'native_router',
    'custom_voices_router',
    'openai_router',
    'profile_router',
    'model_router',
]
