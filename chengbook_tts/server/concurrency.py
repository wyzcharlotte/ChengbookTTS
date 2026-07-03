"""
并发控制
--------
asyncio.Semaphore + ThreadPoolExecutor 实现 GPU 串行安全并发。
"""

import asyncio
import threading
import logging
from typing import Optional
from concurrent.futures import ThreadPoolExecutor

from chengbook_tts.config.settings import settings

_semaphore: Optional[asyncio.Semaphore] = None
_executor: Optional[ThreadPoolExecutor] = None
_pending_count = 0
_pending_lock = threading.Lock()
_total_requests = 0


def init(max_concurrent: int = None):
    """初始化并发控制"""
    global _semaphore, _executor
    max_workers = max_concurrent or settings.MAX_CONCURRENT
    _semaphore = asyncio.Semaphore(max_workers)
    _executor = ThreadPoolExecutor(max_workers=max_workers)
    logging.info(f'Concurrency initialized: max_concurrent={max_workers}')


def shutdown():
    """关闭线程池"""
    if _executor:
        _executor.shutdown(wait=True)


def inc_pending():
    global _pending_count
    with _pending_lock:
        _pending_count += 1


def dec_pending():
    global _pending_count, _total_requests
    with _pending_lock:
        _pending_count -= 1
        _total_requests += 1


def get_pending() -> int:
    return _pending_count


def get_total() -> int:
    return _total_requests


def get_semaphore() -> asyncio.Semaphore:
    assert _semaphore is not None, 'Concurrency not initialized'
    return _semaphore


def get_executor() -> ThreadPoolExecutor:
    assert _executor is not None, 'Concurrency not initialized'
    return _executor
