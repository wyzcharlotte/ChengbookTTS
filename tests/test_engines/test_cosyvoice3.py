"""CosyVoice3 引擎测试（无 GPU，仅测试接口）"""

import pytest
from unittest.mock import MagicMock, patch
import numpy as np
import sys


class TestCosyVoice3Engine:
    """测试 CosyVoice3 引擎接口（不加载真实模型）"""

    @pytest.fixture
    def engine(self, mock_voices_config, mock_emotions_config):
        with patch.dict(sys.modules, {'cosyvoice': MagicMock(), 'cosyvoice.cli': MagicMock(),
                                       'cosyvoice.cli.cosyvoice': MagicMock()}):
            from chengbook_tts.engines.cosyvoice3 import CosyVoice3Engine
            config = {
                'model_dir': '/nonexistent/model',
                'fp16': False,
                'load_trt': False,
            }
            eng = CosyVoice3Engine(mock_voices_config, mock_emotions_config, config)
            return eng

    def test_properties(self, engine):
        assert engine.engine_name == 'CosyVoice3'
        assert engine.supports_streaming is True
        assert engine.supports_emotion is True
        assert engine.supports_multi_speaker is False

    def test_health_info(self, engine):
        info = engine.health_info()
        assert info['engine'] == 'CosyVoice3'
        assert 'sample_rate' in info
