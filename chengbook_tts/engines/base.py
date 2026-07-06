"""
TTS 引擎抽象基类
---------------
所有 TTS 模型（CosyVoice3、SoulX-Podcast、IndexTTS 等）统一实现此接口。
换模型时只需新建一个 Engine 类 + 在 ModelManager 中注册，
服务代码无需任何改动。
"""

import os
import re
import sys
import time
import logging
from abc import ABC, abstractmethod
from typing import Any, Generator, Optional
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
import torch

# 分词器（按需切换，当前使用 jieba）
try:
    import jieba
    _SEG_AVAILABLE = True
except ImportError:
    _SEG_AVAILABLE = False

# PyTorch 全局加速（只执行一次）
try:
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
except Exception:
    pass

try:
    torch.set_float32_matmul_precision('high')
except Exception:
    pass

try:
    torch._C._jit_set_profiling_executor(False)
    torch._C._jit_set_profiling_mode(False)
except Exception:
    pass


# ---------- 中文文本预处理 ----------

_RE_CHINESE_CHAR = re.compile(r'[一-鿿㐀-䶿]')


def _is_chinese_2char(s: str) -> bool:
    """是否为纯中文二字词"""
    return len(s) == 2 and _RE_CHINESE_CHAR.search(s) is not None


def preprocess_chinese_text(text: str, segment: bool = False) -> str:
    """
    中文文本预处理: 相邻二字词自动合并为词组，再用 [词] 包裹。

    例如:
        "我的发票抬头是个人" → "我的 [发票抬头] 是 [个人]"
    """
    if not _SEG_AVAILABLE:
        return text
    if not _RE_CHINESE_CHAR.search(text):
        return text
    if not segment:
        return text

    words = jieba.lcut(text)

    # 第一步: 合并相邻的二字中文词
    merged = []
    buf = []
    for w in words:
        stripped = w.strip()
        if not stripped:
            continue
        if _is_chinese_2char(stripped):
            buf.append(stripped)
        else:
            if buf:
                merged.append(''.join(buf))
                buf.clear()
            merged.append(stripped)
    if buf:
        merged.append(''.join(buf))

    # 第二步: 2-4字词加括号
    result_parts = []
    for w in merged:
        if _RE_CHINESE_CHAR.search(w) is not None and 2 <= len(w) <= 4:
            result_parts.append(f' [{w}] ')
        else:
            result_parts.append(w)

    result = ''.join(result_parts)
    result = re.sub(r' {2,}', ' ', result)
    return result.strip()


# ================================================================
# TTSEngine ABC
# ================================================================

class TTSEngine(ABC):
    """TTS 引擎抽象基类 — 所有模型实现必须继承"""

    def __init__(self, voices_config: dict[str, Any], emotions_config: dict[str, Any],
                 model_config: Optional[dict[str, Any]] = None):
        """
        参数:
            voices_config: 音色配置 {voice_id: {name, wav, description}}
            emotions_config: 情绪配置 {emotion_id: {name, instruct, description}}
            model_config: 模型专属配置（路径、加速选项等）
        """
        self._voices_config: dict[str, Any] = voices_config
        self._emotions_config: dict[str, Any] = emotions_config
        self._model_config: dict[str, Any] = model_config or {}
        self._custom_voice_ids: set = set()
        self._sample_rate: int = 24000
        self._model: Any = None

    # ---- 生命周期 ----

    @abstractmethod
    def initialize(self) -> None:
        """加载模型、预计算缓存、预热。服务启动时调用一次。"""
        ...

    def unload(self) -> None:
        """释放 GPU 内存。切换模型时调用。子类可重写以执行额外清理。"""
        if self._model is not None:
            del self._model
            self._model = None
        torch.cuda.empty_cache()
        import gc
        gc.collect()
        logging.info(f'[{self.engine_name}] Unloaded')

    # ---- 合成接口 ----

    @abstractmethod
    def synthesize(self, text: str, voice: str, emotion: str,
                   speed: float = 1.0, **kwargs) -> NDArray[np.float32]:
        """非流式合成 → 返回完整音频 numpy float32 (1D)"""
        ...

    def synthesize_stream(self, text: str, voice: str, emotion: str,
                          **kwargs) -> Generator[NDArray[np.float32], None, None]:
        """流式合成 → 生成器。默认实现: 调用 synthesize 并 yield 一次"""
        audio = self.synthesize(text, voice, emotion, **kwargs)
        yield audio

    # ---- 音色管理 ----

    def register_voice(self, voice_id: str, wav_path: str, name: str,
                       description: str = '', prompt_text: str = '') -> bool:
        """注册自定义音色（子类应重写）

        参数:
            voice_id: 音色唯一标识
            wav_path: 参考音频路径
            name: 音色显示名称
            description: 音色描述（可选）
            prompt_text: 参考音频对应的转录文本（可选，上传后可提升音色克隆质量）
        """
        raise NotImplementedError(
            f'{self.engine_name} does not support runtime voice registration')

    def unregister_voice(self, voice_id: str) -> bool:
        """注销自定义音色（子类应重写）"""
        raise NotImplementedError(
            f'{self.engine_name} does not support runtime voice unregistration')

    def is_custom_voice(self, voice_id: str) -> bool:
        """判断是否为自定义音色"""
        return voice_id in self._custom_voice_ids

    def voice_info(self, voice_id: str) -> dict[str, Any]:
        """获取音色元信息"""
        return self._voices_config.get(voice_id, {})

    def emotion_info(self, emotion_id: str) -> dict[str, Any]:
        """获取情绪元信息"""
        return self._emotions_config.get(emotion_id, {})

    # ---- 属性 ----

    @property
    @abstractmethod
    def sample_rate(self) -> int:
        """音频采样率"""
        ...

    @property
    @abstractmethod
    def engine_name(self) -> str:
        """引擎名称"""
        ...

    @property
    def engine_version(self) -> str:
        """引擎版本"""
        return self._model_config.get('engine_version', 'unknown')

    @property
    def supports_streaming(self) -> bool:
        """是否支持流式合成"""
        return False

    @property
    def supports_emotion(self) -> bool:
        """是否支持情绪控制"""
        return True

    @property
    def supports_multi_speaker(self) -> bool:
        """是否支持多说话人对话"""
        return False

    @property
    def supports_speed_control(self) -> bool:
        """是否支持语速调节（False 时 UI 灰显语速滑块）"""
        return True

    @property
    def supports_segmentation(self) -> bool:
        """是否支持中文分词（False 时 UI 灰显分词开关）"""
        return True

    @property
    def voice_ids(self) -> list[str]:
        """已加载的音色 ID 列表"""
        return list(self._voices_config.keys())

    @property
    def emotion_ids(self) -> list[str]:
        """已加载的情绪 ID 列表"""
        return list(self._emotions_config.keys())

    @abstractmethod
    def health_info(self) -> dict[str, Any]:
        """返回引擎状态摘要"""
        ...
