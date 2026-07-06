"""
CosyVoice3 引擎实现
------------------
基于 CosyVoice3 模型（LLM + Flow + HiFi-GAN）的 TTS 引擎。
从原 ChengbookTTS_cosyvoice3_ui/engine.py 迁移而来。
"""

import os
import sys
import time
import logging
from typing import Any, Generator

import numpy as np
from numpy.typing import NDArray
import torch

from chengbook_tts.engines.base import TTSEngine, preprocess_chinese_text
from chengbook_tts.config.settings import settings


class CosyVoice3Engine(TTSEngine):
    """基于 CosyVoice3 模型的 TTS 引擎"""

    def __init__(self, voices_config: dict[str, Any], emotions_config: dict[str, Any],
                 model_config: dict[str, Any] = None):
        super().__init__(voices_config, emotions_config, model_config)

        # 模型配置
        self._model_dir = self._model_config.get('model_dir', '')
        self._fp16 = self._model_config.get('fp16', settings.FP16)
        self._load_trt = self._model_config.get('load_trt', settings.LOAD_TRT)
        self._cosyvoice_root = self._model_config.get('cosyvoice_root', '')

        # 内部状态
        self._frontend: Any = None
        self._voice_cache: dict[str, dict[str, Any]] = {}
        self._emotion_cache: dict[str, dict[str, Any]] = {}

    # ---- 属性 ----

    @property
    def sample_rate(self) -> int:
        return self._sample_rate

    @property
    def engine_name(self) -> str:
        return 'CosyVoice3'

    @property
    def engine_version(self) -> str:
        return self._model_config.get('engine_version', '3.0')

    @property
    def supports_streaming(self) -> bool:
        return True

    @property
    def supports_emotion(self) -> bool:
        return True

    @property
    def supports_multi_speaker(self) -> bool:
        return True

    @property
    def voice_ids(self) -> list[str]:
        return list(self._voice_cache.keys())

    @property
    def emotion_ids(self) -> list[str]:
        return list(self._emotion_cache.keys())

    # ---- 生命周期 ----

    def initialize(self) -> None:
        """完整初始化: 路径 → 加载 → 缓存 → 预热"""
        self._setup_path()
        self._load_model()
        self._cache_voices()
        self._cache_emotions()
        self._warmup()
        logging.info(
            f'[CosyVoice3] Ready: {len(self._voice_cache)} voices, '
            f'{len(self._emotion_cache)} emotions, '
            f'sr={self._sample_rate}, fp16={self._fp16}, trt={self._load_trt}'
        )

    def unload(self) -> None:
        """释放 GPU 内存"""
        self._frontend = None
        self._voice_cache.clear()
        self._emotion_cache.clear()
        super().unload()

    def _setup_path(self):
        """注入 cosyvoice + matcha 源码路径"""
        # 优先使用 vendor/ 下的符号链接
        vendor = self._cosyvoice_root or str(settings.VENDOR_DIR)
        sys.path.insert(0, vendor)

        matcha_path = os.path.join(vendor, 'matcha')
        if not os.path.exists(matcha_path):
            matcha_path = os.path.join(vendor, 'third_party', 'Matcha-TTS')
        sys.path.insert(0, matcha_path)

    def _load_model(self):
        """加载 CosyVoice3 模型"""
        from cosyvoice.cli.cosyvoice import AutoModel

        if not self._model_dir:
            raise ValueError(
                'model_dir is empty. Set COSYVOICE3_MODEL_DIR env var '
                'or configure model_configs/cosyvoice3.yaml'
            )

        logging.info(f'[CosyVoice3] Loading from: {self._model_dir}')
        t0 = time.time()

        self._model = AutoModel(
            model_dir=self._model_dir,
            fp16=self._fp16,
            load_trt=self._load_trt,
        )
        self._frontend = self._model.frontend
        self._sample_rate = self._model.sample_rate

        # 诊断：确认模型类型
        print(f'[MODEL-DIAG] AutoModel type={type(self._model).__name__}, '
              f'Model type={type(self._model.model).__name__}, '
              f'model_dir={self._model_dir}', flush=True)

        logging.info(f'[CosyVoice3] Loaded in {time.time() - t0:.1f}s')

    def _cache_voices(self):
        """预计算所有音色的 prompt 特征"""
        logging.info('[CosyVoice3] Pre-computing voice caches ...')
        for voice_id, info in self._voices_config.items():
            wav = info.get('wav', '')
            if not wav or not os.path.exists(wav):
                logging.warning(f'  [{voice_id}] wav not found: {wav}')
                continue

            # 标记自定义音色（custom_ 前缀）
            if voice_id.startswith('custom_'):
                self._custom_voice_ids.add(voice_id)

            t1 = time.time()
            try:
                speech_feat, speech_feat_len = self._frontend._extract_speech_feat(wav)
                speech_token, speech_token_len = self._frontend._extract_speech_token(wav)
                embedding = self._frontend._extract_spk_embedding(wav)
            except Exception as e:
                logging.error(f'  [{voice_id}] extraction failed: {e}')
                continue

            self._voice_cache[voice_id] = {
                'flow_prompt_speech_token': speech_token,
                'flow_prompt_speech_token_len': speech_token_len,
                'prompt_speech_feat': speech_feat,
                'prompt_speech_feat_len': speech_feat_len,
                'llm_embedding': embedding,
                'flow_embedding': embedding,
            }
            logging.info(f'  [{voice_id}] {info["name"]}: cached in {time.time() - t1:.1f}s')

    def _cache_emotions(self):
        """预 tokenize 所有情绪的 instruct 文本"""
        logging.info('[CosyVoice3] Pre-computing emotion caches ...')
        for emotion_id, info in self._emotions_config.items():
            t1 = time.time()
            try:
                tokens, tokens_len = self._frontend._extract_text_token(info['instruct'])
                self._emotion_cache[emotion_id] = {
                    'prompt_text': tokens,
                    'prompt_text_len': tokens_len,
                }
                logging.info(f'  [{emotion_id}] {info["name"]}: cached in {time.time() - t1:.1f}s')
            except Exception as e:
                logging.warning(f'  [{emotion_id}] cache failed: {e}')

    def _warmup(self):
        """预热一次推理"""
        if not self._voice_cache or not self._emotion_cache:
            return
        logging.info('[CosyVoice3] Warming up ...')
        voice_id = list(self._voice_cache.keys())[0]
        emotion_id = list(self._emotion_cache.keys())[0]
        try:
            for _ in self._model.inference_instruct2(
                '测试',
                self._emotions_config[emotion_id]['instruct'],
                self._voices_config[voice_id]['wav'],
                stream=False,
                speed=1.0,
            ):
                pass
            logging.info('[CosyVoice3] Warmup done')
        except Exception as e:
            logging.warning(f'[CosyVoice3] Warmup skipped (non-critical): {e}')

    # ---- 音色管理 ----

    def register_voice(self, voice_id: str, wav_path: str, name: str,
                       description: str = '', prompt_text: str = '') -> bool:
        """运行时注册新音色"""
        if voice_id in self._voice_cache:
            logging.warning(f'[register_voice] {voice_id} already exists')
            return False

        if not os.path.exists(wav_path):
            logging.error(f'[register_voice] wav not found: {wav_path}')
            return False

        t0 = time.time()
        try:
            speech_feat, speech_feat_len = self._frontend._extract_speech_feat(wav_path)
            speech_token, speech_token_len = self._frontend._extract_speech_token(wav_path)
            embedding = self._frontend._extract_spk_embedding(wav_path)
        except Exception as e:
            logging.error(f'[register_voice] {voice_id}: extraction failed: {e}')
            return False

        self._voice_cache[voice_id] = {
            'flow_prompt_speech_token': speech_token,
            'flow_prompt_speech_token_len': speech_token_len,
            'prompt_speech_feat': speech_feat,
            'prompt_speech_feat_len': speech_feat_len,
            'llm_embedding': embedding,
            'flow_embedding': embedding,
        }
        self._voices_config[voice_id] = {
            'name': name,
            'wav': wav_path,
            'description': description or f'自定义音色: {name}',
        }
        self._custom_voice_ids.add(voice_id)
        logging.info(f'[register_voice] {voice_id} "{name}": registered in {time.time() - t0:.1f}s')
        return True

    def unregister_voice(self, voice_id: str) -> bool:
        """运行时注销音色"""
        if voice_id not in self._voice_cache:
            return False
        if voice_id not in self._custom_voice_ids:
            logging.warning(f'[unregister_voice] {voice_id} is preset, cannot unregister')
            return False

        del self._voice_cache[voice_id]
        self._voices_config.pop(voice_id, None)
        self._custom_voice_ids.discard(voice_id)
        logging.info(f'[unregister_voice] {voice_id}: unregistered')
        return True

    # ---- 合成接口 ----

    def synthesize(self, text: str, voice: str, emotion: str,
                   speed: float = 1.0, segment: bool = False, **kwargs) -> NDArray[np.float32]:
        """非流式合成 → numpy float32"""
        model_input = self._build_input(text, voice, emotion, segment=segment)
        print(f'[TTS-DIAG] text_token shape={model_input["text"].shape}, '
              f'prompt_text shape={model_input["prompt_text"].shape}, '
              f'flow_prompt_speech_token shape={model_input["flow_prompt_speech_token"].shape}, '
              f'prompt_speech_feat shape={model_input["prompt_speech_feat"].shape}, '
              f'llm_embedding shape={model_input["llm_embedding"].shape}', flush=True)
        for chunk in self._model.model.tts(**model_input, stream=False, speed=speed):
            result = chunk['tts_speech'].squeeze(0).numpy()
            print(f'[TTS-DIAG] output speech shape={result.shape}, '
                  f'n_samples={result.shape[0]}, duration={result.shape[0]/self.sample_rate:.3f}s', flush=True)
            return result
        raise RuntimeError('synthesize: no output from model')

    def synthesize_stream(self, text: str, voice: str, emotion: str,
                          segment: bool = False, **kwargs) -> Generator[NDArray[np.float32], None, None]:
        """流式合成 → 生成器 yield numpy float32"""
        model_input = self._build_input(text, voice, emotion, segment=segment)
        for chunk in self._model.model.tts(**model_input, stream=True, speed=1.0):
            yield chunk['tts_speech'].squeeze(0).numpy()

    def _build_input(self, text: str, voice: str, emotion: str,
                     segment: bool = True) -> dict[str, Any]:
        """用预缓存 + 请求文本构建 model_input dict"""
        vc = self._voice_cache[voice]
        ec = self._emotion_cache[emotion]

        # 中文分词预处理：per-request segment 覆盖全局
        processed = preprocess_chinese_text(text, segment=segment)
        if processed != text:
            logging.info(f'[Preprocess] "{text}" -> "{processed}"')
        text = processed

        tts_token, tts_token_len = self._frontend._extract_text_token(text)

        return {
            'text': tts_token,
            'text_len': tts_token_len,
            'flow_prompt_speech_token': vc['flow_prompt_speech_token'],
            'flow_prompt_speech_token_len': vc['flow_prompt_speech_token_len'],
            'prompt_speech_feat': vc['prompt_speech_feat'],
            'prompt_speech_feat_len': vc['prompt_speech_feat_len'],
            'llm_embedding': vc['llm_embedding'],
            'flow_embedding': vc['flow_embedding'],
            'prompt_text': ec['prompt_text'],
            'prompt_text_len': ec['prompt_text_len'],
        }

    def health_info(self) -> dict[str, Any]:
        return {
            'engine': self.engine_name,
            'version': self.engine_version,
            'model_dir': self._model_dir,
            'sample_rate': self._sample_rate,
            'fp16': self._fp16,
            'load_trt': self._load_trt,
            'voices_loaded': len(self._voice_cache),
            'emotions_loaded': len(self._emotion_cache),
        }
