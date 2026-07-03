"""voice 命令 — 音色管理"""

import os

from chengbook_tts.engines.manager import ModelManager


def voice_command(args):
    if args.voice_command == 'list':
        _cmd_list()
    elif args.voice_command == 'add':
        _cmd_add(args)
    elif args.voice_command == 'remove':
        _cmd_remove(args)
    else:
        _cmd_list()


def _cmd_list():
    """列出所有音色"""
    manager = ModelManager.get_instance()
    engine = manager.engine

    print(f'\n当前引擎: {engine.engine_name} v{engine.engine_version}')
    print(f'音色数量: {len(engine.voice_ids)}')
    print(f'{"-"*50}')
    for vid in engine.voice_ids:
        info = engine.voice_info(vid)
        custom = '[自定义]' if engine.is_custom_voice(vid) else '[预设]'
        print(f'  {vid:<20} {info.get("name", vid):<16} {custom}')
    print()


def _cmd_add(args):
    """添加自定义音色"""
    manager = ModelManager.get_instance()
    engine = manager.engine

    wav_path = args.wav
    if not os.path.exists(wav_path):
        print(f'错误: WAV 文件不存在: {wav_path}')
        return

    success = engine.register_voice(args.id, wav_path, args.name)
    if success:
        print(f'音色 [{args.id}] "{args.name}" 添加成功!')
    else:
        print(f'音色 [{args.id}] 添加失败（可能已存在或文件无效）')


def _cmd_remove(args):
    """删除自定义音色"""
    manager = ModelManager.get_instance()
    engine = manager.engine

    if args.id not in engine.voice_ids:
        print(f'音色 [{args.id}] 不存在')
        return

    if not engine.is_custom_voice(args.id):
        print(f'音色 [{args.id}] 是预设音色，不可删除')
        return

    success = engine.unregister_voice(args.id)
    if success:
        print(f'音色 [{args.id}] 已删除')
    else:
        print(f'音色 [{args.id}] 删除失败')
