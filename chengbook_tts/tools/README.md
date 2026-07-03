# tools/ — 辅助脚本

独立可执行的辅助工具脚本。

## 文件说明

| 文件 | 说明 |
|------|------|
| `acc2wav.py` | 音频格式转换器 — 支持 AAC、MP3、WAV、FLAC、OGG 等格式互转 |

## 使用示例

```bash
# 单文件转换: AAC → WAV（默认 16kHz 单声道 16bit）
python tools/acc2wav.py input.aac output.wav

# 自定义参数
python tools/acc2wav.py input.aac output.wav -r 22050 -c 2 --normalize

# MP3 → WAV
python tools/acc2wav.py song.mp3 song.wav

# 批量转换
python tools/acc2wav.py --batch ./aac_files ./wav_output --fmt aac --outfmt wav
```

依赖: `pydub` + `ffmpeg`（系统安装）
