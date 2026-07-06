"""
模型管理器
---------
管理模型生命周期: 懒加载、切换、卸载。
同一时刻只允许一个模型驻留在 GPU 显存中。
"""

import os
import gc
import time
import yaml
import logging
import threading
import importlib
from pathlib import Path
from typing import Optional, Any

import torch

from chengbook_tts.engines.base import TTSEngine
from chengbook_tts.config.settings import settings
from chengbook_tts.config.voices import get_voices
from chengbook_tts.config.emotions import get_emotions


class ModelManager:
    """
    模型管理器（单例）
    负责: 懒加载引擎类 → 实例化 → 初始化 → 卸载旧模型
    """

    _instance: Optional['ModelManager'] = None
    _lock = threading.Lock()

    # 模型类型 → (模块路径, 类名)
    ENGINE_CLASS_MAP = {
        'cosyvoice3': ('chengbook_tts.engines.cosyvoice3', 'CosyVoice3Engine'),
        'soulxpodcast': ('chengbook_tts.engines.soulxpodcast', 'SoulXPodcastEngine'),
        'indextts_v1': ('chengbook_tts.engines.indextts_v1', 'IndexTTSEngine'),
        'indextts_v2': ('chengbook_tts.engines.indextts_v2', 'IndexTTS2Engine'),
    }

    def __init__(self):
        self._engine: Optional[TTSEngine] = None
        self._model_type: Optional[str] = None
        self._init_lock = threading.Lock()

    @classmethod
    def get_instance(cls) -> 'ModelManager':
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    # ---- 属性 ----

    @property
    def engine(self) -> TTSEngine:
        if self._engine is None:
            raise RuntimeError('No model loaded. Call load_model() first.')
        return self._engine

    @property
    def model_type(self) -> Optional[str]:
        return self._model_type

    # ---- 模型加载/切换 ----

    def load_model(self, model_type: str) -> TTSEngine:
        """
        加载指定模型。如果已有模型加载，先卸载再加载新模型。

        参数:
            model_type: 'cosyvoice3' | 'soulxpodcast' | 'indextts_v1' | 'indextts_v2'

        返回:
            已初始化的 TTSEngine 实例
        """
        if model_type not in self.ENGINE_CLASS_MAP:
            raise ValueError(
                f'Unknown model type: {model_type}. '
                f'Available: {list(self.ENGINE_CLASS_MAP.keys())}'
            )

        with self._init_lock:
            # 已是目标模型 → 直接返回
            if self._model_type == model_type and self._engine is not None:
                logging.info(f'Model [{model_type}] already loaded')
                return self._engine

            # 卸载当前模型
            if self._engine is not None:
                self._unload_current()

            # 加载目标模型
            logging.info(f'Loading model: [{model_type}] ...')
            t0 = time.time()

            module_path, class_name = self.ENGINE_CLASS_MAP[model_type]
            try:
                module = importlib.import_module(module_path)
                engine_cls = getattr(module, class_name)
            except ImportError as e:
                raise ImportError(
                    f'Failed to import engine for [{model_type}]. '
                    f'Make sure dependencies are installed.\n'
                    f'Module: {module_path}\n'
                    f'Error: {e}'
                )

            # 加载模型配置
            config = self._load_model_config(model_type)

            # 实例化
            engine = engine_cls(
                voices_config=get_voices(include_custom=False),
                emotions_config=get_emotions(),
                model_config=config,
            )

            # 初始化（加载权重、预热）
            engine.initialize()

            self._engine = engine
            self._model_type = model_type

            logging.info(f'Model [{model_type}] loaded in {time.time() - t0:.1f}s')
            return engine

    def _unload_current(self) -> None:
        """卸载当前模型"""
        if self._engine is not None:
            logging.info(f'Unloading model: [{self._model_type}] ...')
            self._engine.unload()
            del self._engine
            self._engine = None
            self._model_type = None
            torch.cuda.empty_cache()
            gc.collect()

    def _load_model_config(self, model_type: str) -> dict[str, Any]:
        """加载模型 YAML 配置"""
        config_dir = Path(__file__).resolve().parent.parent / 'config' / 'model_configs'
        config_path = config_dir / f'{model_type}.yaml'

        if not config_path.exists():
            logging.warning(f'Config not found: {config_path}, using defaults')
            return {'model_type': model_type}

        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)

        # 环境变量覆盖 model_dir
        env_key = f'{model_type.upper()}_MODEL_DIR'
        if os.environ.get(env_key):
            config['model_dir'] = os.environ[env_key]

        # 如果没有配置 model_dir，自动检测
        if not config.get('model_dir'):
            config['model_dir'] = self._auto_detect_model_dir(model_type)

        return config

    def _auto_detect_model_dir(self, model_type: str) -> str:
        """自动检测模型目录"""
        project_root = Path(__file__).resolve().parent.parent.parent

        candidates = {
            'cosyvoice3': [
                project_root / 'models' / 'Fun-CosyVoice3-0.5B',
                settings.VENDOR_DIR / 'cosyvoice',
            ],
            'soulxpodcast': [
                project_root / 'models' / 'SoulX-Podcast-1.7B',
            ],
            'indextts_v1': [
                project_root / 'models' / 'Index-TTS-1.5-vLLM',
                project_root / 'models' / 'Index-TTS-vLLM',
            ],
            'indextts_v2': [
                project_root / 'models' / 'IndexTTS-2-vLLM',
            ],
        }

        for cand in candidates.get(model_type, []):
            if cand.exists():
                return str(cand)

        return ''

    # ---- 模型信息 ----

    def list_models(self) -> list[dict]:
        """列出所有可用模型及其状态（排除 hidden 的模型）"""
        models = []
        for mtype in self.ENGINE_CLASS_MAP:
            config = self._load_model_config(mtype)
            if config.get('hidden', False):
                continue
            model_dir = config.get('model_dir', '')
            is_loaded = self._model_type == mtype

            models.append({
                'type': mtype,
                'name': config.get('engine_name', mtype),
                'version': config.get('engine_version', 'unknown'),
                'model_dir': model_dir,
                'loaded': is_loaded,
                'sample_rate': config.get('sample_rate', 24000),
                'supports_streaming': config.get('supports_streaming', False),
                'supports_emotion': config.get('supports_emotion', True),
                'supports_multi_speaker': config.get('supports_multi_speaker', False),
            })
        return models
