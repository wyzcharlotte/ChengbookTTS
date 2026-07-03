"""
Pytest 配置和共享 fixtures
"""

import pytest
from unittest.mock import MagicMock, patch
import numpy as np


class MockTTSEngine:
    """用于测试的 Mock 引擎"""

    def __init__(self):
        self._sample_rate = 24000
        self._voices = {
            'woman': {'name': '女客户', 'wav': '/mock/woman.wav', 'description': '测试音色'},
            'man': {'name': '男客户', 'wav': '/mock/man.wav', 'description': '测试音色'},
        }
        self._emotions = {
            'calm': {'name': '平和', 'instruct': 'test', 'description': '默认'},
            'happy': {'name': '开心', 'instruct': 'test', 'description': '愉快'},
        }
        self._custom_voice_ids = set()

    @property
    def sample_rate(self):
        return self._sample_rate

    @property
    def engine_name(self):
        return 'MockEngine'

    @property
    def engine_version(self):
        return '1.0.0-test'

    @property
    def supports_streaming(self):
        return True

    @property
    def supports_emotion(self):
        return True

    @property
    def supports_multi_speaker(self):
        return False

    @property
    def voice_ids(self):
        return list(self._voices.keys())

    @property
    def emotion_ids(self):
        return list(self._emotions.keys())

    def initialize(self):
        pass

    def unload(self):
        pass

    def synthesize(self, text, voice, emotion, speed=1.0, **kwargs):
        # 返回固定长度音频
        duration = int(len(text) * 0.05 * self._sample_rate)
        return np.zeros(duration, dtype=np.float32)

    def synthesize_stream(self, text, voice, emotion, **kwargs):
        # 分 3 个 chunk 返回
        total = int(len(text) * 0.05 * self._sample_rate)
        chunk_size = total // 3
        for i in range(3):
            yield np.zeros(chunk_size, dtype=np.float32)

    def voice_info(self, voice_id):
        return self._voices.get(voice_id, {})

    def emotion_info(self, emotion_id):
        return self._emotions.get(emotion_id, {})

    def is_custom_voice(self, voice_id):
        return voice_id in self._custom_voice_ids

    def register_voice(self, voice_id, wav_path, name, description=''):
        self._voices[voice_id] = {'name': name, 'wav': wav_path, 'description': description}
        self._custom_voice_ids.add(voice_id)
        return True

    def unregister_voice(self, voice_id):
        if voice_id in self._custom_voice_ids:
            del self._voices[voice_id]
            self._custom_voice_ids.discard(voice_id)
            return True
        return False

    def health_info(self):
        return {
            'engine': self.engine_name,
            'version': self.engine_version,
            'sample_rate': self._sample_rate,
        }


@pytest.fixture
def mock_engine():
    return MockTTSEngine()


@pytest.fixture
def mock_voices_config():
    return {
        'woman': {'name': '女客户', 'wav': '/mock/woman.wav', 'description': '测试'},
        'man': {'name': '男客户', 'wav': '/mock/man.wav', 'description': '测试'},
    }


@pytest.fixture
def mock_emotions_config():
    return {
        'calm': {'name': '平和', 'instruct': 'test', 'description': '默认'},
        'happy': {'name': '开心', 'instruct': 'test', 'description': '愉快'},
    }
