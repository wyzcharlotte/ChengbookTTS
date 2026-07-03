"""
合成历史记录管理器
-----------------
内存环形缓冲区，记录最近 N 条合成事件。
提供 record() 记录和 get_history() 查询。
"""

import time
import threading
from typing import Optional
from datetime import datetime, timezone, timedelta

# 北京时间时区
TZ_BEIJING = timezone(timedelta(hours=8))

# 最大保留条数
MAX_HISTORY = 200


class SynthesisRecord:
    """单条合成记录"""
    __slots__ = (
        'id', 'timestamp', 'text', 'voice_name', 'emotion_name',
        'speed', 'model', 'duration', 'elapsed', 'rtf',
        'streaming', 'success', 'error',
    )

    def __init__(self, **kwargs):
        self.id: int = kwargs.get('id', 0)
        self.timestamp: str = kwargs.get('timestamp', '')
        self.text: str = kwargs.get('text', '')
        self.voice_name: str = kwargs.get('voice_name', '')
        self.emotion_name: str = kwargs.get('emotion_name', '')
        self.speed: float = kwargs.get('speed', 1.0)
        self.model: str = kwargs.get('model', '')
        self.duration: float = kwargs.get('duration', 0.0)
        self.elapsed: float = kwargs.get('elapsed', 0.0)
        self.rtf: float = kwargs.get('rtf', 0.0)
        self.streaming: bool = kwargs.get('streaming', False)
        self.success: bool = kwargs.get('success', True)
        self.error: str = kwargs.get('error', '')

    def to_dict(self) -> dict:
        return {
            'id': self.id,
            'timestamp': self.timestamp,
            'text': self.text,
            'voice_name': self.voice_name,
            'emotion_name': self.emotion_name,
            'speed': self.speed,
            'model': self.model,
            'duration': round(self.duration, 2),
            'elapsed': round(self.elapsed, 2),
            'rtf': round(self.rtf, 3),
            'streaming': self.streaming,
            'success': self.success,
            'error': self.error,
        }


class HistoryManager:
    """合成历史管理器（线程安全环形缓冲区）"""

    def __init__(self, max_size: int = MAX_HISTORY):
        self._max_size = max_size
        self._records: list[SynthesisRecord] = []
        self._next_id: int = 1
        self._lock = threading.Lock()

    def record(
        self,
        text: str,
        voice_name: str = '',
        emotion_name: str = '',
        speed: float = 1.0,
        model: str = '',
        duration: float = 0.0,
        elapsed: float = 0.0,
        rtf: float = 0.0,
        streaming: bool = False,
        success: bool = True,
        error: str = '',
    ) -> SynthesisRecord:
        """记录一条合成事件，返回记录对象"""
        now = datetime.now(TZ_BEIJING).strftime('%H:%M:%S')
        rec = SynthesisRecord(
            id=self._next_id,
            timestamp=now,
            text=text[:30] + ('...' if len(text) > 30 else ''),
            voice_name=voice_name,
            emotion_name=emotion_name,
            speed=speed,
            model=model,
            duration=duration,
            elapsed=elapsed,
            rtf=rtf,
            streaming=streaming,
            success=success,
            error=error,
        )

        with self._lock:
            self._next_id += 1
            self._records.append(rec)
            # 环形 — 超出上限时丢弃最旧的
            if len(self._records) > self._max_size:
                self._records = self._records[-self._max_size:]

        return rec

    def get_history(self, limit: int = 50, offset: int = 0) -> dict:
        """获取历史记录（倒序，最新的在前）"""
        with self._lock:
            all_records = list(reversed(self._records))
            total = len(all_records)
            page = all_records[offset:offset + limit]
            return {
                'total': total,
                'limit': limit,
                'offset': offset,
                'records': [r.to_dict() for r in page],
            }

    def clear(self):
        """清空历史"""
        with self._lock:
            self._records.clear()

    @property
    def count(self) -> int:
        with self._lock:
            return len(self._records)


# 全局单例
_history_manager: Optional[HistoryManager] = None
_lock = threading.Lock()


def get_history_manager() -> HistoryManager:
    global _history_manager
    if _history_manager is None:
        with _lock:
            if _history_manager is None:
                _history_manager = HistoryManager()
    return _history_manager
