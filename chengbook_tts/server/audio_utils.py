"""
音频格式转换工具
---------------
numpy array ↔ WAV bytes ↔ ffmpeg 转码。
从原 server.py 迁移而来。
"""

import os
import io
import subprocess
import tempfile
import logging

import numpy as np
import soundfile as sf

SUPPORTED_FORMATS = {'mp3', 'opus', 'aac', 'flac', 'wav', 'pcm'}

FORMAT_MIME = {
    'mp3': 'audio/mpeg',
    'opus': 'audio/opus',
    'aac': 'audio/aac',
    'flac': 'audio/flac',
    'wav': 'audio/wav',
    'pcm': 'audio/L16',
}


def audio_to_bytes(audio: np.ndarray, sample_rate: int, fmt: str = 'wav') -> bytes:
    """
    将 numpy float32 音频转换为指定格式的 bytes。

    参数:
        audio: float32 numpy 数组 (1D)
        sample_rate: 采样率
        fmt: 目标格式 (mp3/opus/aac/flac/wav/pcm)

    返回:
        音频 bytes
    """
    if fmt == 'pcm':
        return (audio * 32767).astype(np.int16).tobytes()

    if fmt not in SUPPORTED_FORMATS:
        raise ValueError(f'Unsupported format: {fmt}. Supported: {SUPPORTED_FORMATS}')

    # 先写 WAV，再用 ffmpeg 转码
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_wav:
        wav_path = tmp_wav.name

    try:
        sf.write(wav_path, audio, sample_rate, format='WAV')

        if fmt == 'wav':
            with open(wav_path, 'rb') as f:
                return f.read()

        out_path = wav_path + f'.{fmt}'
        codec_map = {
            'mp3': 'libmp3lame',
            'opus': 'libopus',
            'aac': 'aac',
            'flac': 'flac',
        }
        codec = codec_map.get(fmt, 'libmp3lame')
        subprocess.run([
            'ffmpeg', '-y', '-i', wav_path,
            '-c:a', codec,
            '-b:a', '128k' if fmt == 'mp3' else '64k',
            out_path,
        ], capture_output=True, check=True)

        with open(out_path, 'rb') as f:
            return f.read()
    finally:
        _cleanup(wav_path)
        if fmt != 'wav':
            _cleanup(wav_path + f'.{fmt}')


def audio_to_wav_bytes(audio: np.ndarray, sample_rate: int) -> bytes:
    """将音频 numpy 数组编码为 WAV bytes"""
    buf = io.BytesIO()
    sf.write(buf, audio, sample_rate, format='WAV')
    buf.seek(0)
    return buf.read()


def _cleanup(path: str):
    try:
        os.unlink(path)
    except OSError:
        pass
