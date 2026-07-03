"""
IndexTTS v1/v1.5 引擎适配器
--------------------------
将 IndexTTS (GPT + BigVGAN) 模型封装为 TTSEngine 接口。
支持 Index-TTS 1.0 和 Index-TTS 1.5 两个版本。
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


class IndexTTSEngine(TTSEngine):
    """IndexTTS v1/v1.5 引擎适配器 (GPT + BigVGAN)"""

    def __init__(self, voices_config: dict[str, Any], emotions_config: dict[str, Any],
                 model_config: dict[str, Any] = None):
        super().__init__(voices_config, emotions_config, model_config)

        self._model_dir = self._model_config.get('model_dir', '')
        self._gpu_memory_utilization = self._model_config.get('gpu_memory_utilization', 0.25)
        self._tts: Any = None
        self._speaker_registry: dict[str, Any] = {}
        self._voice_wavs: dict[str, str] = {}

    # ---- 属性 ----

    @property
    def sample_rate(self) -> int:
        return self._sample_rate

    @property
    def engine_name(self) -> str:
        return 'IndexTTS'

    @property
    def engine_version(self) -> str:
        return self._model_config.get('engine_version', '1.5')

    @property
    def supports_streaming(self) -> bool:
        return False

    @property
    def supports_emotion(self) -> bool:
        return False  # v1 不支持情绪控制

    @property
    def supports_multi_speaker(self) -> bool:
        return False

    # ---- 生命周期 ----

    def initialize(self) -> None:
        """加载 IndexTTS 模型"""
        self._setup_path()

        from indextts.infer_vllm import IndexTTS

        if not self._model_dir:
            raise ValueError(
                'model_dir is empty. Set INDEXTTS_V1_MODEL_DIR env var '
                'or configure model_configs/indextts_v1.yaml'
            )

        logging.info(f'[IndexTTS] Loading from: {self._model_dir}')
        t0 = time.time()

        self._tts = IndexTTS(
            model_dir=self._model_dir,
            gpu_memory_utilization=self._gpu_memory_utilization,
        )
        self._sample_rate = 24000

        # 预注册预设音色
        for voice_id, info in self._voices_config.items():
            wav = info.get('wav', '')
            if wav and os.path.exists(wav):
                self._register_speaker(voice_id, wav, info.get('name', voice_id))

        logging.info(
            f'[IndexTTS] Loaded in {time.time() - t0:.1f}s, '
            f'voices={len(self._voice_wavs)}'
        )

    def unload(self) -> None:
        """释放 GPU 内存"""
        self._tts = None
        self._speaker_registry.clear()
        self._voice_wavs.clear()
        super().unload()

    def _setup_path(self):
        """注入 IndexTTS 源码路径"""
        indextts_root = os.path.join(
            str(settings.ROOT_DIR.parent.parent), 'Index-TTS-Vllm2', 'index-tts-vllm'
        )
        if indextts_root not in sys.path:
            sys.path.insert(0, indextts_root)

    def _register_speaker(self, voice_id: str, wav_path: str, name: str):
        """注册说话人到 IndexTTS"""
        try:
            self._tts.registry_speaker(voice_id, [wav_path])
            self._speaker_registry[voice_id] = wav_path
            self._voice_wavs[voice_id] = wav_path
            logging.info(f'  [{voice_id}] {name}: registered')
        except Exception as e:
            logging.warning(f'  [{voice_id}] {name}: registration failed: {e}')

    # ---- 合成接口 ----

    def synthesize(self, text: str, voice: str, emotion: str = 'calm',
                   speed: float = 1.0, **kwargs) -> NDArray[np.float32]:
        """
        语音合成。IndexTTS v1 不支持 emotion 参数，会被忽略。
        """
        if voice not in self._speaker_registry:
            raise ValueError(
                f'Voice not found: {voice}. Available: {list(self._speaker_registry.keys())}'
            )

        import asyncio

        # IndexTTS 的 infer_with_ref_audio_embed 是 async 方法
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        sr, wav = loop.run_until_complete(
            self._tts.infer_with_ref_audio_embed(voice, text)
        )

        # IndexTTS 输出格式: int16 range numpy array
        # 转换为 float32 归一化
        if wav.dtype == np.int16:
            wav = wav.astype(np.float32) / 32767.0
        else:
            wav = wav.astype(np.float32)

        # 确保是 1D array
        if wav.ndim > 1:
            wav = wav.squeeze()

        return wav

    # ---- 音色管理 ----

    def register_voice(self, voice_id: str, wav_path: str, name: str,
                       description: str = '') -> bool:
        if voice_id in self._speaker_registry:
            return False
        if not os.path.exists(wav_path):
            return False

        self._register_speaker(voice_id, wav_path, name)
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
        self._speaker_registry.pop(voice_id, None)
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
            'voices_loaded': len(self._speaker_registry),
        }
