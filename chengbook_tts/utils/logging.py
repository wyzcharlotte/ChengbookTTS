"""
工具函数 — 日志配置
-------------------
统一的日志格式和级别控制。
"""

import logging
import os


def setup_logging(name: str = 'chengbook_tts',
                  level: str = None) -> logging.Logger:
    """
    配置统一的日志格式。

    参数:
        name: logger 名称
        level: 日志级别 (DEBUG/INFO/WARNING/ERROR)，默认从环境变量 LOG_LEVEL 读取
    """
    if level is None:
        level = os.environ.get('LOG_LEVEL', 'INFO').upper()

    log_level = getattr(logging, level, logging.INFO)
    fmt = logging.Formatter(
        '[%(asctime)s] [%(levelname)s] %(name)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
    )

    # 配置 root logger，确保所有 logging.info() 调用有输出
    root = logging.getLogger()
    root.setLevel(log_level)
    if not root.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(fmt)
        root.addHandler(handler)

    logger = logging.getLogger(name)
    return logger
