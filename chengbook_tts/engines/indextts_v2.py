"""
IndexTTS-2 引擎适配器
--------------------
将 IndexTTS-2 (GPT + w2v-bert-2.0 + s2mel + BigVGAN + QwenEmotion) 封装为 TTSEngine。
支持情绪控制（emo_vector、emo_text）。
"""

import os
import sys
import gc
import time
import logging
from typing import Any, Optional

import numpy as np
from numpy.typing import NDArray
import torch

from chengbook_tts.engines.base import TTSEngine
from chengbook_tts.config.settings import settings


class IndexTTS2Engine(TTSEngine):
    """IndexTTS-2 引擎适配器 (GPT + w2v-bert + s2mel + BigVGAN + QwenEmotion)"""

    def __init__(self, voices_config: dict[str, Any], emotions_config: dict[str, Any],
                 model_config: dict[str, Any] = None):
        super().__init__(voices_config, emotions_config, model_config)

        self._model_dir = self._model_config.get('model_dir', '')
        self._gpu_memory_utilization = self._model_config.get('gpu_memory_utilization', 0.25)
        self._qwenemo_gpu_memory_utilization = self._model_config.get(
            'qwenemo_gpu_memory_utilization', 0.10
        )
        self._tts: Any = None
        self._voice_wavs: dict[str, str] = {}

    # ---- 属性 ----

    @property
    def sample_rate(self) -> int:
        return self._sample_rate

    @property
    def engine_name(self) -> str:
        return 'IndexTTS2'

    @property
    def engine_version(self) -> str:
        return self._model_config.get('engine_version', '2.0')

    @property
    def supports_streaming(self) -> bool:
        return False

    @property
    def supports_emotion(self) -> bool:
        return True  # v2 支持情绪控制

    @property
    def supports_multi_speaker(self) -> bool:
        return False

    # ---- 生命周期 ----

    def initialize(self) -> None:
        """加载 IndexTTS-2 模型"""
        self._setup_path()

        from indextts.infer_vllm_v2 import IndexTTS2

        if not self._model_dir:
            raise ValueError(
                'model_dir is empty. Set INDEXTTS_V2_MODEL_DIR env var '
                'or configure model_configs/indextts_v2.yaml'
            )

        logging.info(f'[IndexTTS2] Loading from: {self._model_dir}')
        t0 = time.time()

        self._tts = IndexTTS2(
            model_dir=self._model_dir,
            gpu_memory_utilization=self._gpu_memory_utilization,
            qwenemo_gpu_memory_utilization=self._qwenemo_gpu_memory_utilization,
        )
        self._sample_rate = 22050  # IndexTTS-2 输出 22.05kHz

        # 缓存音色路径
        for voice_id, info in self._voices_config.items():
            wav = info.get('wav', '')
            if wav and os.path.exists(wav):
                self._voice_wavs[voice_id] = wav

        logging.info(
            f'[IndexTTS2] Loaded in {time.time() - t0:.1f}s, '
            f'voices={len(self._voice_wavs)}'
        )

    def unload(self) -> None:
        """释放 GPU 内存"""
        self._tts = None
        self._voice_wavs.clear()
        super().unload()

    def _setup_path(self):
        """注入 IndexTTS 源码路径"""
        indextts_root = os.path.join(
            str(settings.ROOT_DIR.parent.parent), 'Index-TTS-Vllm2', 'index-tts-vllm'
        )
        if indextts_root not in sys.path:
            sys.path.insert(0, indextts_root)

    # ---- 情绪向量映射 ----

    def _emotion_to_vector(self, emotion: str) -> Optional[list[float]]:
        """
        将通用情绪名映射到 IndexTTS-2 的 8 维情绪向量。

        IndexTTS-2 情绪向量格式:
        [happy, angry, sad, fear, hate, low, surprise, neutral]
        每个元素 0.0-1.4，总和建议 <= 1.5
        """
        mapping = {
            'calm':      [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],  # neutral
            'happy':     [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # happy
            'sad':       [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # sad
            'angry':     [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # angry
            'soft':      [0.0, 0.0, 0.0, 0.0, 0.0, 0.6, 0.0, 0.4],  # low + neutral
            'loud':      [0.0, 0.8, 0.0, 0.0, 0.0, 0.0, 0.2, 0.0],  # angry + surprise
            'impatient': [0.0, 0.5, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0],  # angry + hate
            'confused':  [0.0, 0.0, 0.0, 0.3, 0.0, 0.0, 0.7, 0.0],  # fear + surprise
        }
        return mapping.get(emotion)

    # ---- 合成接口 ----

    def synthesize(self, text: str, voice: str, emotion: str = 'calm',
                   speed: float = 1.0, **kwargs) -> NDArray[np.float32]:
        """
        语音合成。v2 支持情绪控制（emo_vector 方式）。
        """
        if voice not in self._voice_wavs:
            raise ValueError(
                f'Voice not found: {voice}. Available: {list(self._voice_wavs.keys())}'
            )

        import asyncio

        # 情绪 → 向量
        emo_vector = self._emotion_to_vector(emotion)

        wav_path = self._voice_wavs[voice]

        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        sr, wav = loop.run_until_complete(
            self._tts.infer(
                spk_audio_prompt=wav_path,
                text=text,
                emo_vector=emo_vector,
                use_emo_text=False,
                emo_text=None,
            )
        )

        # IndexTTS-2 输出: int16
        if wav.dtype == np.int16:
            wav = wav.astype(np.float32) / 32767.0
        else:
            wav = wav.astype(np.float32)

        if wav.ndim > 1:
            wav = wav.squeeze()

        return wav

    # ---- 音色管理 ----

    def register_voice(self, voice_id: str, wav_path: str, name: str,
                       description: str = '') -> bool:
        if voice_id in self._voice_wavs:
            return False
        if not os.path.exists(wav_path):
            return False

        self._voice_wavs[voice_id] = wav_path
        self._voices_config[voice_id] = {
            'name': name,
            'wav': wav_path,
            'description': description or f'自定义音色: {name}',
        }
        self._custom_voice_ids.add(voice_id)
        return True

    def unregister_voice(self, voice_id: str) -> bool:
        if voice_id not in self._custom_voice_ids:
            return False
        self._voice_wavs.pop(voice_id, None)
        self._voices_config.pop(voice_id, None)
        self._custom_voice_ids.discard(voice_id)
        return True

    def health_info(self) -> dict[str, Any]:
        return {
            'engine': self.engine_name,
            'version': self.engine_version,
            'model_dir': self._model_dir,
            'gpu_memory_utilization': self._gpu_memory_utilization,
            'sample_rate': self._sample_rate,
            'voices_loaded': len(self._voice_wavs),
        }
