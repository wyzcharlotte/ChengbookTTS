"""
SoulX-Podcast 引擎适配器
-----------------------
将 SoulX-Podcast 多说话人对播模型封装为 TTSEngine 接口。
"""

import os
import sys
import gc
import time
import logging
from typing import Any

import numpy as np
from numpy.typing import NDArray
import torch

from chengbook_tts.engines.base import TTSEngine
from chengbook_tts.config.settings import settings
from chengbook_tts.engines.soulxpodcast_humanize import (
    apply_humanization, perturb_sampling_params, build_config,
)


class SoulXPodcastEngine(TTSEngine):
    """SoulX-Podcast 1.7B 引擎适配器"""

    def __init__(self, voices_config: dict[str, Any], emotions_config: dict[str, Any],
                 model_config: dict[str, Any] = None):
        super().__init__(voices_config, emotions_config, model_config)

        self._model_dir = self._model_config.get('model_dir', '')
        self._llm_engine = self._model_config.get('llm_engine', 'hf')
        self._fp16_flow = self._model_config.get('fp16_flow', False)
        self._tokenizer_version = self._model_config.get('tokenizer_version', 'speech_tokenizer_v2_25hz')

        self._dataset: Any = None
        self._voice_wavs: dict[str, str] = {}

    # ---- 属性 ----

    @property
    def sample_rate(self) -> int:
        return self._sample_rate

    @property
    def engine_name(self) -> str:
        return 'SoulXPodcast'

    @property
    def engine_version(self) -> str:
        return self._model_config.get('engine_version', '1.7B')

    @property
    def supports_streaming(self) -> bool:
        return False

    @property
    def supports_emotion(self) -> bool:
        return False  # TODO: SoulX-Podcast 情绪映射待完善

    @property
    def supports_multi_speaker(self) -> bool:
        return True  # 共享音色克隆库

    @property
    def supports_speed_control(self) -> bool:
        return False

    @property
    def supports_segmentation(self) -> bool:
        return False

    # ---- 生命周期 ----

    def initialize(self) -> None:
        """加载 SoulX-Podcast 模型"""
        self._setup_path()

        # s3tokenizer 依赖真正的 onnx 包，但 CosyVoice3 的 matcha vendor 里有个
        # onnx/ 目录遮蔽了它。临时移除 matcha 路径后再导入 onnx。
        _matcha_paths = [p for p in sys.path if 'matcha' in p.lower() and 'onnx' in os.listdir(p)]
        for mp in _matcha_paths:
            sys.path.remove(mp)
        import onnx
        for mp in reversed(_matcha_paths):
            sys.path.insert(0, mp)

        from soulxpodcast.models.soulxpodcast import SoulXPodcast
        from soulxpodcast.config import Config, SoulXPodcastLLMConfig
        from soulxpodcast.utils.dataloader import PodcastInferHandler

        # Monkey-patch s3tokenizer.load_audio: 绕过 torchcodec (PyTorch 2.9+cu128 不兼容)
        self._patch_s3tokenizer_load_audio()

        if not self._model_dir:
            raise ValueError(
                'model_dir is empty. Set SOULXPODCAST_MODEL_DIR env var '
                'or configure model_configs/soulxpodcast.yaml'
            )

        logging.info(f'[SoulXPodcast] Loading from: {self._model_dir}')
        t0 = time.time()

        # 加载配置
        hf_config = SoulXPodcastLLMConfig.from_initial_and_json(
            initial_values={'fp16_flow': self._fp16_flow},
            json_file=os.path.join(self._model_dir, 'soulxpodcast_config.json'),
        )
        self._config = Config(
            model=self._model_dir,
            enforce_eager=False,  # Phase 1b: enable CUDA graphs
            llm_engine=self._llm_engine,
            hf_config=hf_config,
        )

        # 加载模型
        self._model = SoulXPodcast(self._config)
        self._dataset = PodcastInferHandler(
            self._model.llm.tokenizer, None, self._config
        )
        self._sample_rate = 24000

        # 用 model config 中定义的音色（SoulX-Podcast 示例音频 + 准确转录文本），
        # 兜底回退到全局 voices_config。
        model_voices = self._model_config.get('voices', {})
        prompts_dir = str(settings.PROMPTS_DIR / 'soulxpodcast')
        for voice_id, info in self._voices_config.items():
            mv = model_voices.get(voice_id, {})
            wav = mv.get('wav', '') or info.get('wav', '')
            # 如果是空字符串或相对路径，解析为 prompts/soulxpodcast 下的音频
            if not wav or not os.path.isabs(wav):
                if voice_id == 'woman':
                    wav = os.path.join(prompts_dir, 'woman.wav')
                elif voice_id == 'man':
                    wav = os.path.join(prompts_dir, 'man.wav')
            prompt_text = mv.get('prompt_text', '') or info.get('prompt_text', '')
            if wav and os.path.exists(wav):
                self._voice_wavs[voice_id] = wav
                # 标记自定义音色
                if voice_id.startswith('custom_'):
                    self._custom_voice_ids.add(voice_id)
                if prompt_text:
                    self._voices_config[voice_id] = {**info, 'wav': wav, 'prompt_text': prompt_text}
                else:
                    self._voices_config[voice_id] = {**info, 'wav': wav}

        logging.info(
            f'[SoulXPodcast] Loaded in {time.time() - t0:.1f}s, '
            f'llm_engine={self._llm_engine}, voices={len(self._voice_wavs)}'
        )

    def unload(self) -> None:
        """释放 GPU 内存"""
        self._dataset = None
        self._voice_wavs.clear()
        super().unload()

    def _setup_path(self):
        """注入 vendor/ 源码路径（soulxpodcast 包）"""
        vendor = str(settings.VENDOR_DIR)
        if vendor not in sys.path:
            sys.path.insert(0, vendor)

    @staticmethod
    def _patch_s3tokenizer_load_audio():
        """用 soundfile 替换 s3tokenizer 内部的 torchaudio.load，绕过 torchcodec"""
        import s3tokenizer
        import soundfile as sf

        def _patched_load_audio(file: str, sr: int = 16000):
            import torch
            import torchaudio
            audio_np, sample_rate = sf.read(file, dtype='float32')
            if audio_np.ndim > 1:
                audio_np = audio_np[:, 0]
            audio = torch.from_numpy(audio_np.copy()).float()  # [T]
            if sample_rate != sr:
                audio = torchaudio.transforms.Resample(sample_rate, sr)(audio.unsqueeze(0)).squeeze(0)
            return audio

        s3tokenizer.load_audio = _patched_load_audio
        s3tokenizer.utils.load_audio = _patched_load_audio

    # ---- 合成接口 ----

    def synthesize(self, text: str, voice: str, emotion: str,
                   speed: float = 1.0, segment: bool = True,
                   **kwargs) -> NDArray[np.float32]:
        """
        单说话人语音合成。

        SoulX-Podcast 原生支持多说话人对播，这里将其简化为
        单说话人模式（[S1] 前缀 + 一段文本）。
        """
        import s3tokenizer
        from soulxpodcast.config import SamplingParams

        if voice not in self._voice_wavs:
            raise ValueError(
                f'Voice not found: {voice}. Available: {list(self._voice_wavs.keys())}'
            )

        # ---- 拟人化文本预处理 ----
        humanize_config = self._build_humanize_config(kwargs)
        original_text = text
        text = apply_humanization(text, humanize_config)
        if humanize_config.enabled:
            changed = text != original_text
            flag = '★' if changed else '→(无变化)'
            logging.info(f'[Humanize] {flag} 预处理前: {original_text}')
            logging.info(f'[Humanize] {flag} 预处理后: {text}')
            print(f'[Humanize] {flag} 预处理前: {original_text}', flush=True)
            print(f'[Humanize] {flag} 预处理后: {text}', flush=True)

        wav_path = self._voice_wavs[voice]
        prompt_text = self._voices_config.get(voice, {}).get('prompt_text', '')

        # 情绪 → 采样参数映射
        sampling_params = self._emotion_to_params(emotion, speed)

        # ---- 采样参数随机扰动 ----
        sampling_params = perturb_sampling_params(sampling_params, humanize_config)

        # 构造单说话人对播数据
        dataitem = {
            'key': 'req_001',
            'prompt_text': [prompt_text],
            'prompt_wav': [wav_path],
            'text': [text],
            'spk': [0],
        }
        self._dataset.update_datasource([dataitem])
        data = self._dataset[0]

        # ---- Phase 1d: 动态生成参数 ----
        # data['text_tokens'] is list[list[int]] (one per turn); sum all turns.
        # 25Hz speech tokenizer, ~6 tokens/char. Use ~20x for 3x headroom.
        text_tokens_count = sum(len(t) for t in data['text_tokens'])
        sampling_params.max_tokens = min(3000, max(600, text_tokens_count * 20))
        sampling_params.min_tokens = 4

        # 准备输入
        prompt_mels_llm, prompt_mels_lens_llm = s3tokenizer.padding(data['log_mel'])
        spk_emb = torch.tensor(data['spk_emb'])
        prompt_mels_flow = torch.nn.utils.rnn.pad_sequence(
            data['mel'], batch_first=True, padding_value=0
        )
        prompt_mels_lens_flow = torch.tensor(data['mel_len'])

        processed = {
            'prompt_mels_for_llm': prompt_mels_llm,
            'prompt_mels_lens_for_llm': prompt_mels_lens_llm,
            'prompt_text_tokens_for_llm': data['prompt_text_tokens'],
            'text_tokens_for_llm': data['text_tokens'],
            'prompt_mels_for_flow_ori': prompt_mels_flow,
            'prompt_mels_lens_for_flow': prompt_mels_lens_flow,
            'spk_emb_for_flow': spk_emb,
            'sampling_params': sampling_params,
            'spk_ids': data['spks_list'],
            'infos': [data['info']],
            'use_dialect_prompt': False,
        }

        # 推理
        with torch.no_grad():
            results = self._model.forward_longform(**processed)

        # 合并音频
        audio = None
        for wav in results['generated_wavs']:
            audio = wav if audio is None else torch.cat([audio, wav], dim=1)

        return audio.cpu().squeeze(0).numpy().astype(np.float32)

    def _emotion_to_params(self, emotion: str, speed: float) -> Any:
        """将通用情绪名映射到 SoulX-Podcast 采样参数

        v2: 微调 temperature/top_k/top_p 让不同情绪有更明显的区分度。
        use_ras=False 保持与上游 transformers 最大兼容性。
        """
        from soulxpodcast.config import SamplingParams

        emotion_map = {
            'calm':     SamplingParams(temperature=0.6, top_k=100, top_p=0.9,  use_ras=True),
            'sad':      SamplingParams(temperature=0.5, top_k=80,  top_p=0.85, use_ras=True),
            'soft':     SamplingParams(temperature=0.5, top_k=80,  top_p=0.85, use_ras=True),
            'confused': SamplingParams(temperature=0.55, top_k=90, top_p=0.9,  use_ras=True),

            'happy':    SamplingParams(temperature=0.8, top_k=120, top_p=0.95, use_ras=True),

            'angry':    SamplingParams(temperature=0.9, top_k=150, top_p=0.95, use_ras=True),
            'loud':     SamplingParams(temperature=0.9, top_k=150, top_p=0.95, use_ras=True),
            'impatient': SamplingParams(temperature=0.85, top_k=130, top_p=0.95, use_ras=True),
        }
        return emotion_map.get(emotion, emotion_map['calm'])

    def _build_humanize_config(self, kwargs: dict[str, Any]):
        """从 API kwargs 构建拟人化配置"""
        humanize = kwargs.get('humanize', False)
        level = kwargs.get('humanize_level', 'moderate')
        return build_config(humanize, level)

    # ---- 音色管理 ----

    def register_voice(self, voice_id: str, wav_path: str, name: str,
                       description: str = '', prompt_text: str = '') -> bool:
        if voice_id in self._voice_wavs:
            return False
        if not os.path.exists(wav_path):
            return False

        self._voice_wavs[voice_id] = wav_path
        self._voices_config[voice_id] = {
            'name': name,
            'wav': wav_path,
            'description': description or f'自定义音色: {name}',
            'prompt_text': prompt_text,
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
            'llm_engine': self._llm_engine,
            'sample_rate': self._sample_rate,
            'voices_loaded': len(self._voice_wavs),
            'emotions_loaded': len(self._emotions_config),
        }
