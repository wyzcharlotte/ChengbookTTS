"""
FastAPI 应用工厂
---------------
创建 FastAPI 实例，配置 lifespan、中间件、路由。
"""

import logging
import sys
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from chengbook_tts.engines.manager import ModelManager
from chengbook_tts.server import concurrency
from chengbook_tts.server.console import serve_console
from chengbook_tts.server.routes import (
    native_router,
    custom_voices_router,
    openai_router,
    profile_router,
    model_router,
)
from chengbook_tts.server.routes.custom_voices import load_custom_voices_on_startup
from chengbook_tts.server.routes.history import router as history_router
from chengbook_tts.server.dependencies import get_engine
from chengbook_tts.config.settings import settings


def create_app() -> FastAPI:
    """创建并配置 FastAPI 应用"""

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        # ---- 启动 ----
        concurrency.init(settings.MAX_CONCURRENT)

        # 初始化模型
        model_type = settings.MODEL_TYPE
        manager = ModelManager.get_instance()
        logging.info(f'Starting with model: [{model_type}]')
        try:
            manager.load_model(model_type)
        except Exception as e:
            logging.error(f'Failed to load model [{model_type}]: {e}')
            # 尝试回退到 cosyvoice3
            if model_type != 'cosyvoice3':
                logging.info('Falling back to cosyvoice3...')
                try:
                    manager.load_model('cosyvoice3')
                except Exception as e2:
                    logging.error(f'Fallback also failed: {e2}')
                    raise
            else:
                raise

        engine = get_engine()
        logging.info(
            f'Service ready: {engine.engine_name} v{engine.engine_version}, '
            f'{len(engine.voice_ids)} voices, {len(engine.emotion_ids)} emotions'
        )

        # 恢复自定义音色
        try:
            load_custom_voices_on_startup()
        except Exception as e:
            logging.warning(f'Custom voice restore failed (non-critical): {e}')

        yield

        # ---- 关闭 ----
        logging.info('Service shutting down')
        concurrency.shutdown()

    app = FastAPI(
        title='ChengbookTTS — 统一多模型 TTS 平台',
        version='2.0.0',
        lifespan=lifespan,
    )

    # CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=['*'],
        allow_credentials=True,
        allow_methods=['*'],
        allow_headers=['*'],
    )

    # 路由
    app.include_router(native_router, prefix='/api')
    app.include_router(custom_voices_router, prefix='/api')
    app.include_router(profile_router, prefix='/api')
    app.include_router(model_router, prefix='/api')
    app.include_router(history_router, prefix='/api')
    app.include_router(openai_router, prefix='/v1')

    # Web 控制台
    app.get('/')(serve_console)

    return app
