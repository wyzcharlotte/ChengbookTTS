# cli/ — 命令行工具

统一命令行接口，支持模型管理、服务启动、直接合成、音色管理。

## 使用方式

```bash
# 入口（任选其一）
python -m chengbook_tts.cli [COMMAND]
chengbook-tts [COMMAND]  # pip install 后
```

## 命令列表

### 模型管理
```bash
# 列出所有可用模型及状态
python -m chengbook_tts.cli model list

# 查看模型详情
python -m chengbook_tts.cli model info cosyvoice3

# 切换当前模型（卸载当前 → 加载新模型）
python -m chengbook_tts.cli model switch soulxpodcast
```

### 服务管理
```bash
# 启动 TTS 服务（默认 CosyVoice3，端口 8080）
python -m chengbook_tts.cli serve

# 指定模型和端口
python -m chengbook_tts.cli serve --model indextts_v2 --port 6006

# 指定 host
python -m chengbook_tts.cli serve --host 0.0.0.0 --port 8080
```

### 直接合成
```bash
# 一行命令合成语音
python -m chengbook_tts.cli synth --text "你好世界" --voice woman --emotion calm

# 指定输出路径
python -m chengbook_tts.cli synth --text "测试" --output output.wav --model cosyvoice3

# 调语速
python -m chengbook_tts.cli synth --text "快点说" --speed 1.5
```

### 音色管理
```bash
# 列出音色
python -m chengbook_tts.cli voice list

# 添加自定义音色
python -m chengbook_tts.cli voice add --id my_voice --wav path/to/voice.wav --name "我的音色"

# 删除自定义音色
python -m chengbook_tts.cli voice remove --id my_voice
```

## 实现文件

| 文件 | 说明 |
|------|------|
| `__main__.py` | `python -m` 入口 |
| `main.py` | argparse 分发器 |
| `commands/serve.py` | 服务启动命令 |
| `commands/model.py` | 模型管理命令 |
| `commands/synth.py` | 直接合成命令 |
| `commands/voice.py` | 音色管理命令 |
