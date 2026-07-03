"""synth 命令 — 直接语音合成（不需要启动服务）"""

import os
import time
import logging

import numpy as np


def synth_command(args):
    """命令行直接合成语音"""
    if args.model:
        os.environ['MODEL_TYPE'] = args.model

    from chengbook_tts.engines.manager import ModelManager
    from chengbook_tts.utils.audio import save_wav
    from chengbook_tts.config.settings import settings

    manager = ModelManager.get_instance()
    model_type = args.model or settings.MODEL_TYPE

    print(f'加载模型 [{model_type}] ...')
    engine = manager.load_model(model_type)
    print(f'引擎: {engine.engine_name} v{engine.engine_version}')

    # 校验音色和情绪
    if args.voice not in engine.voice_ids:
        print(f'错误: 音色 [{args.voice}] 不可用。可选: {engine.voice_ids}')
        return
    if args.emotion not in engine.emotion_ids:
        print(f'错误: 情绪 [{args.emotion}] 不可用。可选: {engine.emotion_ids}')
        return

    print(f'合成中: voice={args.voice}, emotion={args.emotion}, speed={args.speed}')
    print(f'文本: {args.text[:80]}{"..." if len(args.text) > 80 else ""}')

    t0 = time.time()
    try:
        audio = engine.synthesize(args.text, args.voice, args.emotion, args.speed)
    except Exception as e:
        print(f'合成失败: {e}')
        return

    elapsed = time.time() - t0
    duration = len(audio) / engine.sample_rate if engine.sample_rate > 0 else 0
    print(f'完成! 时长: {duration:.2f}s, 耗时: {elapsed:.2f}s, RTF: {elapsed/duration:.3f}')

    # 保存
    output_path = args.output or f'output_{model_type}.wav'
    save_wav(audio, engine.sample_rate, __import__('pathlib').Path(output_path))
    print(f'音频已保存到: {output_path}')
