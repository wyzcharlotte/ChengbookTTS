"""
引擎工厂
--------
create_engine() 根据配置自动检测并创建对应的引擎实例。
支持 auto 检测和手动指定。
"""

import os
import logging
from pathlib import Path
from typing import Optional, Any

from chengbook_tts.engines.base import TTSEngine
from chengbook_tts.engines.manager import ModelManager
from chengbook_tts.config.settings import settings


def create_engine(model_type: str = 'auto',
                  model_dir: Optional[str] = None,
                  voices_config: Optional[dict] = None,
                  emotions_config: Optional[dict] = None) -> TTSEngine:
    """
    TTS 引擎工厂 — 根据模型类型创建对应的引擎实例。

    参数:
        model_type: 模型类型，可选:
            'auto'         — 自动检测（从环境变量或模型目录）
            'cosyvoice3'   — CosyVoice3
            'soulxpodcast' — SoulX-Podcast 1.7B
            'indextts_v1'  — IndexTTS v1/v1.5
            'indextts_v2'  — IndexTTS-2
        model_dir: 模型目录路径（可选，覆盖配置）
        voices_config: 音色配置（可选）
        emotions_config: 情绪配置（可选）

    返回:
        已初始化的 TTSEngine 实例
    """
    if model_type == 'auto':
        model_type = _detect_model_type(model_dir)

    # 如果指定了 model_dir，临时设置环境变量
    if model_dir:
        os.environ[f'{model_type.upper()}_MODEL_DIR'] = model_dir

    manager = ModelManager.get_instance()
    return manager.load_model(model_type)


def _detect_model_type(model_dir: Optional[str] = None) -> str:
    """
    自动检测模型类型。

    检测顺序:
    1. MODEL_TYPE 环境变量
    2. 检查模型目录中的配置文件 (cosyvoice3.yaml, cosyvoice2.yaml 等)
    3. 默认 cosyvoice3
    """
    # 1. 环境变量
    env_type = os.environ.get('MODEL_TYPE', '')
    if env_type and env_type != 'auto':
        logging.info(f'[Factory] MODEL_TYPE from env: {env_type}')
        return env_type

    # 2. 检查模型目录
    model_dir = model_dir or os.environ.get('MODEL_DIR', '')
    if model_dir and os.path.exists(model_dir):
        # CosyVoice 检测
        if os.path.exists(os.path.join(model_dir, 'cosyvoice3.yaml')):
            return 'cosyvoice3'
        if os.path.exists(os.path.join(model_dir, 'cosyvoice2.yaml')):
            return 'cosyvoice3'

        # SoulX-Podcast 检测
        if os.path.exists(os.path.join(model_dir, 'soulxpodcast_config.json')):
            return 'soulxpodcast'

        # IndexTTS 检测
        if os.path.exists(os.path.join(model_dir, 'config.yaml')):
            if 'IndexTTS-2' in model_dir or 'IndexTTS2' in model_dir:
                return 'indextts_v2'
            return 'indextts_v1'

        logging.warning(
            f'[Factory] Cannot detect model type from directory: {model_dir}\n'
            f'Falling back to cosyvoice3'
        )

    # 3. 默认
    return settings.MODEL_TYPE if settings.MODEL_TYPE != 'auto' else 'cosyvoice3'
