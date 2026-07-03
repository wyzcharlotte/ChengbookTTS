"""
工具函数 — 音频处理
------------------
音频格式转换、归一化、静音裁剪、重采样等。
"""

import io
import logging
import numpy as np
from numpy.typing import NDArray
from pathlib import Path


def normalize_audio(audio: NDArray[np.float32], target_peak: float = 0.95) -> NDArray[np.float32]:
    """峰值归一化到 target_peak"""
    peak = np.max(np.abs(audio))
    if peak > 0:
        return audio * (target_peak / peak)
    return audio


def trim_silence(audio: NDArray[np.float32], threshold: float = 0.01,
                 sample_rate: int = 24000, min_silence_ms: int = 200) -> NDArray[np.float32]:
    """裁剪首尾静音"""
    frame_len = int(sample_rate * min_silence_ms / 1000)
    if frame_len <= 0:
        return audio

    # 找首个超过阈值的帧
    energy = np.abs(audio)
    for i in range(0, len(audio) - frame_len, frame_len):
        if np.mean(energy[i:i + frame_len]) > threshold:
            break
    start = i

    # 找末尾最后超过阈值的帧
    for i in range(len(audio) - frame_len, 0, -frame_len):
        if np.mean(energy[i:i + frame_len]) > threshold:
            break
    end = min(i + frame_len, len(audio))

    if end <= start:
        return audio
    return audio[start:end]


def save_wav(audio: NDArray[np.float32], sample_rate: int, output_path: Path) -> None:
    """保存 float32 音频为 WAV 文件"""
    import soundfile as sf
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(output_path), audio, sample_rate)


def load_wav(file_path: Path, target_sr: int = 24000) -> tuple[NDArray[np.float32], int]:
    """加载 WAV 文件，可重采样"""
    import soundfile as sf
    try:
        audio, sr = sf.read(str(file_path), dtype='float32')
    except Exception:
        import librosa
        audio, sr = librosa.load(str(file_path), sr=None, mono=True)
        audio = audio.astype(np.float32)

    if sr != target_sr:
        import librosa
        audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
        sr = target_sr

    return audio, sr
