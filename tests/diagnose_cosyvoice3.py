"""
Diagnostic script: test CosyVoice3 synthesis directly.
"""
import sys
import os
import time
import logging
import numpy as np

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s [%(levelname)s] %(message)s')

# Add project path
sys.path.insert(0, r'e:\program\ChengbookTTS')

from chengbook_tts.config.settings import settings
from chengbook_tts.config.voices import get_voices
from chengbook_tts.config.emotions import get_emotions
from chengbook_tts.engines.cosyvoice3 import CosyVoice3Engine

# Load config
import yaml
from pathlib import Path
config_dir = Path(r'e:\program\ChengbookTTS\chengbook_tts\config\model_configs')
with open(config_dir / 'cosyvoice3.yaml', 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)

# Auto-detect model dir
project_root = Path(r'e:\program\ChengbookTTS').parent
candidate = project_root / 'ChengbookTTS_cosyvoice3' / 'models' / 'Fun-CosyVoice3-0.5B'
if candidate.exists():
    config['model_dir'] = str(candidate)
    print(f'Model dir: {config["model_dir"]}')
else:
    print(f'ERROR: Model dir not found: {candidate}')
    sys.exit(1)

# Create engine
voices = get_voices()
emotions = get_emotions()
print(f'Voices: {list(voices.keys())}')
print(f'Emotions: {list(emotions.keys())}')
for vid, v in voices.items():
    print(f'  {vid}: wav={v["wav"]} exists={os.path.exists(v["wav"])}')
for eid, e in emotions.items():
    print(f'  {eid}: instruct={e["instruct"][:60]}...')

print('\n--- Initializing engine ---')
engine = CosyVoice3Engine(voices, emotions, config)
engine.initialize()

print(f'\nSample rate: {engine.sample_rate}')
print(f'Voices loaded: {engine.voice_ids}')
print(f'Emotions loaded: {engine.emotion_ids}')

# Test 1: segment=False (current default)
print('\n=== Test 1: segment=False ===')
text = '你好，我想查一下我的订单到哪了'
print(f'Text: {text}')
t0 = time.time()
audio = engine.synthesize(text, 'woman', 'calm', speed=1.0, segment=False)
elapsed = time.time() - t0
print(f'Shape: {audio.shape}, dtype: {audio.dtype}')
print(f'Duration: {len(audio)/engine.sample_rate:.3f}s')
print(f'Min: {audio.min():.6f}, Max: {audio.max():.6f}, Mean: {audio.mean():.6f}')
print(f'Non-zero samples: {np.count_nonzero(audio)}/{len(audio)}')
print(f'Elapsed: {elapsed:.2f}s')

# Test 2: segment=True
print('\n=== Test 2: segment=True ===')
t0 = time.time()
audio2 = engine.synthesize(text, 'woman', 'calm', speed=1.0, segment=True)
elapsed = time.time() - t0
print(f'Shape: {audio2.shape}, dtype: {audio2.dtype}')
print(f'Duration: {len(audio2)/engine.sample_rate:.3f}s')
print(f'Min: {audio2.min():.6f}, Max: {audio2.max():.6f}, Mean: {audio2.mean():.6f}')
print(f'Non-zero samples: {np.count_nonzero(audio2)}/{len(audio2)}')
print(f'Elapsed: {elapsed:.2f}s')

# Save both for inspection
import soundfile as sf
sf.write('/tmp/test_segment_false.wav', audio, engine.sample_rate)
sf.write('/tmp/test_segment_true.wav', audio2, engine.sample_rate)
print('\nSaved to /tmp/test_segment_false.wav and /tmp/test_segment_true.wav')

# Test 3: Using inference_instruct2 directly
print('\n=== Test 3: inference_instruct2 (direct) ===')
t0 = time.time()
for chunk in engine._model.inference_instruct2(
    text,
    emotions['calm']['instruct'],
    voices['woman']['wav'],
    stream=False,
    speed=1.0,
):
    audio3 = chunk['tts_speech'].squeeze(0).numpy()
elapsed = time.time() - t0
print(f'Shape: {audio3.shape}, dtype: {audio3.dtype}')
print(f'Duration: {len(audio3)/engine.sample_rate:.3f}s')
print(f'Min: {audio3.min():.6f}, Max: {audio3.max():.6f}, Mean: {audio3.mean():.6f}')
print(f'Elapsed: {elapsed:.2f}s')
sf.write('/tmp/test_instruct2.wav', audio3, engine.sample_rate)
print('Saved to /tmp/test_instruct2.wav')

print('\n=== DONE ===')
