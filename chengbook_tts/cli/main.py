"""
CLI 主入口
---------
argparse 分发器，支持 model/serve/synth/voice 四个子命令。
"""

import sys
import argparse

from chengbook_tts.utils.logging import setup_logging


def main():
    setup_logging('chengbook_tts')

    parser = argparse.ArgumentParser(
        prog='chengbook-tts',
        description='ChengbookTTS — 统一多模型 TTS 平台',
    )
    subparsers = parser.add_subparsers(dest='command', help='子命令')

    # ---- serve ----
    serve_parser = subparsers.add_parser('serve', help='启动 TTS 服务')
    serve_parser.add_argument('--model', type=str, default=None,
                              help='模型类型 (cosyvoice3/soulxpodcast/indextts_v1/indextts_v2)')
    serve_parser.add_argument('--host', type=str, default='0.0.0.0', help='监听地址')
    serve_parser.add_argument('--port', type=int, default=8080, help='监听端口')

    # ---- model ----
    model_parser = subparsers.add_parser('model', help='模型管理')
    model_sub = model_parser.add_subparsers(dest='model_command')

    model_sub.add_parser('list', help='列出所有可用模型')
    info_parser = model_sub.add_parser('info', help='查看模型详情')
    info_parser.add_argument('name', type=str, nargs='?', help='模型类型名')
    switch_parser = model_sub.add_parser('switch', help='切换模型')
    switch_parser.add_argument('name', type=str, help='模型类型名')

    # ---- synth ----
    synth_parser = subparsers.add_parser('synth', help='直接合成语音')
    synth_parser.add_argument('--text', type=str, required=True, help='待合成文本')
    synth_parser.add_argument('--voice', type=str, default='woman', help='音色 ID')
    synth_parser.add_argument('--emotion', type=str, default='calm', help='情绪 ID')
    synth_parser.add_argument('--speed', type=float, default=1.0, help='语速')
    synth_parser.add_argument('--model', type=str, default=None, help='模型类型')
    synth_parser.add_argument('--output', type=str, default=None, help='输出文件路径')

    # ---- voice ----
    voice_parser = subparsers.add_parser('voice', help='音色管理')
    voice_sub = voice_parser.add_subparsers(dest='voice_command')

    voice_sub.add_parser('list', help='列出所有音色')
    add_parser = voice_sub.add_parser('add', help='添加自定义音色')
    add_parser.add_argument('--id', type=str, required=True, help='音色 ID')
    add_parser.add_argument('--wav', type=str, required=True, help='WAV 文件路径')
    add_parser.add_argument('--name', type=str, required=True, help='音色名称')
    remove_parser = voice_sub.add_parser('remove', help='删除自定义音色')
    remove_parser.add_argument('--id', type=str, required=True, help='音色 ID')

    args = parser.parse_args()

    if args.command == 'serve':
        from chengbook_tts.cli.commands.serve import serve_command
        serve_command(args)
    elif args.command == 'model':
        from chengbook_tts.cli.commands.model import model_command
        model_command(args)
    elif args.command == 'synth':
        from chengbook_tts.cli.commands.synth import synth_command
        synth_command(args)
    elif args.command == 'voice':
        from chengbook_tts.cli.commands.voice import voice_command
        voice_command(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == '__main__':
    main()
