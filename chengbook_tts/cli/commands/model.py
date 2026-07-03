"""model 命令 — 模型管理 (list/info/switch)"""

import os
import logging

from chengbook_tts.engines.manager import ModelManager


def model_command(args):
    if args.model_command == 'list':
        _cmd_list()
    elif args.model_command == 'info':
        _cmd_info(args.name)
    elif args.model_command == 'switch':
        _cmd_switch(args.name)
    else:
        _cmd_list()


def _cmd_list():
    """列出所有可用模型"""
    manager = ModelManager.get_instance()
    models = manager.list_models()

    print(f'\n{"="*60}')
    print(f'  可用模型 ({len(models)} 个)')
    print(f'{"="*60}')
    print(f'  {"模型":<20} {"版本":<10} {"采样率":<8} {"状态"}')
    print(f'  {"-"*56}')

    for m in models:
        status = '[当前]' if m['loaded'] else '[未加载]'
        print(f'  {m["type"]:<20} {m["version"]:<10} {m["sample_rate"]}Hz    {status}')

    print(f'{"="*60}')
    print()
    print('切换模型: python -m chengbook_tts.cli model switch <name>')


def _cmd_info(name: str = None):
    """查看模型详情"""
    manager = ModelManager.get_instance()
    models = manager.list_models()

    if name:
        models = [m for m in models if m['type'] == name]

    if not models:
        print(f'模型 [{name}] 未找到')
        return

    for m in models:
        print(f'\n模型: {m["name"]} ({m["type"]})')
        print(f'  版本: {m["version"]}')
        print(f'  采样率: {m["sample_rate"]}Hz')
        print(f'  模型目录: {m["model_dir"]}')
        print(f'  流式合成: {"✅" if m["supports_streaming"] else "❌"}')
        print(f'  情绪控制: {"✅" if m["supports_emotion"] else "❌"}')
        print(f'  多说话人: {"✅" if m["supports_multi_speaker"] else "❌"}')
        print(f'  状态: {"已加载" if m["loaded"] else "未加载"}')


def _cmd_switch(name: str):
    """切换模型"""
    print(f'正在切换到模型 [{name}] ...')
    manager = ModelManager.get_instance()
    try:
        engine = manager.load_model(name)
        print(f'切换成功!')
        print(f'  引擎: {engine.engine_name} v{engine.engine_version}')
        print(f'  采样率: {engine.sample_rate}Hz')
        print(f'  音色: {engine.voice_ids}')
        print(f'  情绪: {engine.emotion_ids}')
    except Exception as e:
        print(f'切换失败: {e}')
