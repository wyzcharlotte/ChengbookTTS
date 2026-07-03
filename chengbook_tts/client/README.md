# client/ — Python SDK

ChengbookTTS 的 Python 客户端库，支持同步调用、流式播放和交互式测试。

## 文件说明

| 文件 | 说明 |
|------|------|
| `sdk.py` | `ChengbookTTSClient` — 封装所有 API 的 Python SDK |
| `streaming.py` | 流式客户端 — 实时 PCM 播放（sounddevice） |
| `interactive.py` | 交互式测试客户端 — 全功能 CLI 测试工具 |

## SDK 使用示例

```python
from chengbook_tts.client import ChengbookTTSClient

# 连接到本地服务
client = ChengbookTTSClient(base_url="http://localhost:8080")

# 健康检查
health = client.health()
print(health)

# 非流式合成
audio = client.tts(text="你好世界", voice="woman", emotion="calm")

# 获取音色/情绪列表
voices = client.list_voices()
emotions = client.list_emotions()

# 自定义音色
client.register_custom_voice(
    voice_id="my_voice",
    wav_path="path/to/voice.wav",
    name="我的音色"
)
```
