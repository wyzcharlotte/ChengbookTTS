# config/model_configs/ — 模型 YAML 配置

每个模型一个 YAML 文件，定义模型路径、加速选项和能力标志。

## 文件列表

| 文件 | 模型 | 引擎类 |
|------|------|--------|
| `cosyvoice3.yaml` | CosyVoice3 | `CosyVoice3Engine` |
| `soulxpodcast.yaml` | SoulX-Podcast 1.7B | `SoulXPodcastEngine` |
| `indextts_v1.yaml` | IndexTTS v1/v1.5 | `IndexTTSEngine` |
| `indextts_v2.yaml` | IndexTTS-2 | `IndexTTS2Engine` |

## YAML 格式

```yaml
model_type: cosyvoice3          # 模型标识（与 MODEL_TYPE 环境变量对应）
model_dir: ""                   # 模型权重目录（留空自动检测）
engine_name: "CosyVoice3"       # 引擎显示名
engine_version: "3.0"           # 版本号
sample_rate: 24000              # 音频采样率

# 模型专属配置...
supports_streaming: true        # 是否支持流式合成
supports_emotion: true          # 是否支持情绪控制
supports_multi_speaker: false   # 是否支持多说话人
```

## 环境变量覆盖

以下环境变量可覆盖 YAML 中的 `model_dir`：

- `COSYVOICE3_MODEL_DIR` → cosyvoice3
- `SOULXPODCAST_MODEL_DIR` → soulxpodcast
- `INDEXTTS_V1_MODEL_DIR` → indextts_v1
- `INDEXTTS_V2_MODEL_DIR` → indextts_v2
