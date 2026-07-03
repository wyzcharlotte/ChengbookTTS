# config/ — 配置中心

统一管理 ChengbookTTS 的所有配置。

## 文件说明

| 文件 | 说明 |
|------|------|
| `settings.py` | 全局设置 dataclass — 路径、服务端口、并发数、默认值等。所有配置可通过环境变量覆盖。 |
| `voices.py` | 音色预设 — `VOICES` dict 定义预设音色（woman/man），`get_voices()` 合并自定义音色。 |
| `emotions.py` | 情绪预设 — `EMOTIONS` dict 定义 8 种通用情绪及其 instruct 文本。 |
| `model_configs/` | 各模型的 YAML 配置文件（模型路径、加速选项、能力标志等）。 |

## 配置优先级

1. 环境变量（最高优先级）
2. YAML 配置文件（`model_configs/*.yaml`）
3. `settings.py` 中的默认值（最低优先级）

## 模型切换

通过环境变量 `MODEL_TYPE` 或 CLI `--model` 参数选择模型：

```bash
# 方式 1: 环境变量
export MODEL_TYPE=soulxpodcast

# 方式 2: CLI 参数
python -m chengbook_tts.cli serve --model indextts_v2
```

## 添加新音色

在 `voices.py` 的 `VOICES` dict 中添加条目，将 WAV 文件放入 `prompts/` 对应目录：

```python
'new_voice': {
    'name': '新音色',
    'wav': str(PROMPTS_BASE / 'new_voice.wav'),
    'description': '这是新音色的描述',
}
```
