"""
端到端诊断：模拟 server 的完整合成→编码→返回流程，检查输出 WAV。
在 WSL2 终端运行：
  python tests/e2e_audio_diag.py
"""
import sys, io, time, struct, os, logging
import numpy as np
import soundfile as sf

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from chengbook_tts.config.settings import settings
from chengbook_tts.config.voices import get_voices
from chengbook_tts.config.emotions import get_emotions
from chengbook_tts.engines.cosyvoice3 import CosyVoice3Engine
from pathlib import Path
import yaml

# --- 1. 加载引擎 ---
config_dir = Path(__file__).resolve().parent.parent / 'chengbook_tts' / 'config' / 'model_configs'
with open(config_dir / 'cosyvoice3.yaml', 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)

project_root = Path(__file__).resolve().parent.parent.parent
candidate = project_root / 'ChengbookTTS_cosyvoice3' / 'models' / 'Fun-CosyVoice3-0.5B'
config['model_dir'] = str(candidate)

voices = get_voices()
emotions = get_emotions()
engine = CosyVoice3Engine(voices, emotions, config)
engine.initialize()

text = '你好，我想查一下订单'
voice = 'woman'
emotion = 'calm'

print(f'\n{"="*60}')
print(f'Engine: {engine.engine_name} v{engine.engine_version}')
print(f'Sample rate: {engine.sample_rate}')
print(f'Voices: {engine.voice_ids}')
print(f'Emotions: {engine.emotion_ids}')
print(f'{"="*60}')

# --- 2. 模拟 server TTS route 的完整流程 ---
print('\n[1] Calling engine.synthesize()...')
t0 = time.time()
audio = engine.synthesize(text, voice, emotion, speed=1.0, segment=False)
elapsed = time.time() - t0

print(f'  Audio shape: {audio.shape}')
print(f'  Audio dtype: {audio.dtype}')
print(f'  Duration: {len(audio)/engine.sample_rate:.3f}s')
print(f'  Amplitude: min={audio.min():.4f}, max={audio.max():.4f}, mean={abs(audio).mean():.6f}')
print(f'  Non-zero: {np.count_nonzero(audio)}/{len(audio)}')
print(f'  Synthesis time: {elapsed:.2f}s')

# --- 3. 用 BytesIO + soundfile 编码（完全模拟 server） ---
print('\n[2] Encoding with soundfile to BytesIO...')
buf = io.BytesIO()
sf.write(buf, audio, engine.sample_rate, format='WAV')
buf.seek(0)
wav_bytes = buf.read()

print(f'  WAV size: {len(wav_bytes)} bytes')

# --- 4. 验证 WAV 头 ---
print('\n[3] Analyzing WAV header...')
if len(wav_bytes) >= 44:
    riff = wav_bytes[:4]
    file_size = struct.unpack('<I', wav_bytes[4:8])[0]
    wave = wav_bytes[8:12]
    fmt_mark = wav_bytes[12:16]
    fmt_size = struct.unpack('<I', wav_bytes[16:20])[0]
    audio_fmt = struct.unpack('<H', wav_bytes[20:22])[0]
    channels = struct.unpack('<H', wav_bytes[22:24])[0]
    sr = struct.unpack('<I', wav_bytes[24:28])[0]
    byte_rate = struct.unpack('<I', wav_bytes[28:32])[0]
    block_align = struct.unpack('<H', wav_bytes[32:34])[0]
    bits_per_sample = struct.unpack('<H', wav_bytes[34:36])[0]

    fmt_names = {1: 'PCM (int)', 3: 'IEEE float', 6: 'A-law', 7: 'Mu-law', 0xFFFE: 'Extensible'}
    fmt_name = fmt_names.get(audio_fmt, f'Unknown({audio_fmt})')

    print(f'  RIFF: {riff.decode()}')
    print(f'  WAVE: {wave.decode()}')
    print(f'  File size: {file_size} (+8 = {file_size+8})')
    print(f'  Audio format: {audio_fmt} = {fmt_name}')
    print(f'  Channels: {channels}')
    print(f'  Sample rate: {sr}')
    print(f'  Byte rate: {byte_rate}')
    print(f'  Block align: {block_align}')
    print(f'  Bits per sample: {bits_per_sample}')

    # 找到 data chunk
    data_pos = wav_bytes.find(b'data')
    if data_pos > 0:
        data_size = struct.unpack('<I', wav_bytes[data_pos+4:data_pos+8])[0]
        print(f'  Data chunk at offset {data_pos}, size={data_size} bytes')
        expected = len(audio) * (bits_per_sample // 8) * channels
        print(f'  Expected data size: {expected} bytes')

        if audio_fmt == 3:  # Float WAV
            audio_data = np.frombuffer(wav_bytes[data_pos+8:data_pos+8+data_size], dtype=np.float32)
        else:
            audio_data = np.frombuffer(wav_bytes[data_pos+8:data_pos+8+data_size], dtype=np.int16)
        print(f'  Samples read from WAV: {len(audio_data)}')
        print(f'  Amplitude: min={audio_data.min():.4f}, max={audio_data.max():.4f}')
else:
    print(f'  WAV too short: {len(wav_bytes)} bytes!')

# --- 5. 保存到文件 ---
out_path = '/tmp/e2e_test_output.wav'
with open(out_path, 'wb') as f:
    f.write(wav_bytes)
print(f'\n[4] Saved to {out_path}')

# --- 6. 尝试用 soundfile 读回来验证 ---
print('\n[5] Roundtrip verification:')
try:
    data_back, sr_back = sf.read(out_path)
    print(f'  Read back: shape={data_back.shape}, sr={sr_back}, dtype={data_back.dtype}')
    print(f'  Match original: shape_match={data_back.shape == audio.shape}, '
          f'sr_match={sr_back == engine.sample_rate}')
except Exception as e:
    print(f'  Read back FAILED: {e}')

print(f'\n{"="*60}')
print('Diagnostics complete. Try playing: /tmp/e2e_test_output.wav')
print(f'{"="*60}')
