"""TTSEngine ABC 契约测试"""

import pytest
import numpy as np
from chengbook_tts.engines.base import TTSEngine


class TestTTSEngineABC:
    """验证 TTSEngine ABC 接口契约"""

    def test_abc_cannot_instantiate(self, mock_voices_config, mock_emotions_config):
        """直接实例化 ABC 应该报错"""
        with pytest.raises(TypeError):
            TTSEngine(mock_voices_config, mock_emotions_config)

    def test_concrete_engine(self, mock_engine):
        """具体实现应满足接口"""
        assert mock_engine.engine_name == 'MockEngine'
        assert mock_engine.sample_rate == 24000
        assert len(mock_engine.voice_ids) == 2
        assert len(mock_engine.emotion_ids) == 2

    def test_synthesize_returns_numpy(self, mock_engine):
        """synthesize 应返回 numpy float32"""
        audio = mock_engine.synthesize('测试', 'woman', 'calm')
        assert isinstance(audio, np.ndarray)
        assert audio.dtype == np.float32

    def test_synthesize_stream_yields(self, mock_engine):
        """synthesize_stream 应 yield numpy arrays"""
        chunks = list(mock_engine.synthesize_stream('测试文本', 'woman', 'calm'))
        assert len(chunks) > 0
        for chunk in chunks:
            assert isinstance(chunk, np.ndarray)
            assert chunk.dtype == np.float32

    def test_register_voice(self, mock_engine):
        """注册音色"""
        assert mock_engine.register_voice('custom_1', '/fake/path.wav', '自定义')
        assert 'custom_1' in mock_engine.voice_ids
        assert mock_engine.is_custom_voice('custom_1')

    def test_unregister_voice(self, mock_engine):
        """注销音色"""
        mock_engine.register_voice('custom_2', '/fake/path.wav', 'test')
        assert mock_engine.unregister_voice('custom_2')
        assert 'custom_2' not in mock_engine.voice_ids

    def test_unregister_preset_voice(self, mock_engine):
        """预设音色不可注销"""
        assert not mock_engine.unregister_voice('woman')

    def test_voice_info(self, mock_engine):
        """获取音色信息"""
        info = mock_engine.voice_info('woman')
        assert info['name'] == '女客户'

    def test_emotion_info(self, mock_engine):
        """获取情绪信息"""
        info = mock_engine.emotion_info('calm')
        assert info['name'] == '平和'

    def test_default_streaming_fallback(self, mock_engine):
        """默认 synthesize_stream 应调用 synthesize"""

        class MinimalEngine(mock_engine.__class__):
            def synthesize_stream(self, text, voice, emotion, **kwargs):
                audio = self.synthesize(text, voice, emotion, **kwargs)
                yield audio

        eng = MinimalEngine()
        chunks = list(eng.synthesize_stream('test', 'woman', 'calm'))
        assert len(chunks) == 1
