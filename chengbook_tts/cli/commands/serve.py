"""serve 命令 — 启动 TTS 服务"""

import os
import logging


def serve_command(args):
    """启动 FastAPI 服务"""
    if args.model:
        os.environ['MODEL_TYPE'] = args.model

    host = args.host or '0.0.0.0'
    port = args.port or 8080

    from chengbook_tts.server.app import create_app
    import uvicorn

    app = create_app()
    logging.info(f'Starting server on {host}:{port}')
    uvicorn.run(app, host=host, port=port, log_level='info')
