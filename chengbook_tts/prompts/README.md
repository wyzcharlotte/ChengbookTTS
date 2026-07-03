# prompts/ — 音色提示音频

存放 TTS 音色克隆所需的提示音频（WAV 格式）。

## 目录说明

| 目录 | 说明 |
|------|------|
| `cosyvoice3/` | 符号链接 → `ChengbookTTS_cosyvoice3_ui/Prompt`（预设音色） |
| `custom/` | 用户通过 API 上传的自定义音色 WAV 文件 |

## 音色克隆流程

1. **预设音色**: 将 WAV 文件放入对应目录，在 `config/voices.py` 的 `VOICES` dict 中注册
2. **自定义音色**: 通过 API `POST /api/voices/custom` 上传，系统自动提取音色特征

## 音频文件要求

- **格式**: WAV（推荐 16kHz/24kHz 单声道）
- **时长**: 3~10 秒（过短特征不足，过长影响速度）
- **内容**: 清晰的人声，无背景噪音，单说话人
- **大小**: 上传限制 10 MB

## 音色切换

通过 `config/voices.py` 的 `VOICES` dict 或 API `/api/profile` 切换当前音色。
也可以通过 CLI：

```bash
python -m chengbook_tts.cli synth --text "你好" --voice woman
```
