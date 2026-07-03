# ChengbookTTS — 统一多模型 TTS 平台

集成了 **CosyVoice3**、**SoulX-Podcast**、**Index-TTS** 三大 TTS 模型，通过统一的 API 和 CLI 提供语音合成服务。

## 支持的模型

| 模型 | 引擎类 | 采样率 | 流式 | 情绪控制 | 多说话人 |
|------|--------|--------|------|----------|----------|
| CosyVoice3 | `CosyVoice3Engine` | 24000 | ✅ | ✅ (instruct) | ❌ |
| SoulX-Podcast 1.7B | `SoulXPodcastEngine` | 24000 | ❌ | ✅ (采样参数) | ✅ |
| IndexTTS v1/1.5 | `IndexTTSEngine` | 24000 | ❌ | ❌ | ❌ |
| IndexTTS-2 | `IndexTTS2Engine` | 22050 | ❌ | ✅ (情绪向量) | ❌ |

## 快速开始

```bash
# 安装依赖
pip install -r requirements.txt

# 启动服务（默认 CosyVoice3）
python -m chengbook_tts.cli serve --port 8080

# 切换模型
python -m chengbook_tts.cli model switch soulxpodcast

# 命令行合成
python -m chengbook_tts.cli synth --text "你好世界" --voice woman --model cosyvoice3
```

## 项目结构

```
ChengbookTTS/
├── chengbook_tts/           # 主包
│   ├── config/              # 统一配置
│   ├── engines/             # 引擎抽象层
│   ├── server/              # FastAPI 服务
│   ├── cli/                 # 命令行工具
│   ├── client/              # Python SDK
│   ├── vendor/              # 模型源码（符号链接）
│   ├── prompts/             # 音色提示音频
│   ├── utils/               # 工具函数
│   └── tools/               # 辅助脚本
├── tests/                   # 测试套件
├── scripts/                 # 启动/安装脚本
└── docs/                    # 文档
```

## API 端点

### 原生 API (`/api/*`)
- `GET /api/health` — 健康检查
- `GET /api/voices` — 音色列表
- `GET /api/emotions` — 情绪列表
- `POST /api/tts` — 非流式合成
- `POST /api/tts/stream` — 流式合成
- `POST /api/voices/custom` — 注册自定义音色
- `DELETE /api/voices/custom/{voice_id}` — 删除自定义音色
- `GET /api/profile` / `POST /api/profile` — 配置文件管理

### OpenAI 兼容 API (`/v1/*`)
- `GET /v1/models` — 模型列表
- `GET /v1/health` — 健康检查
- `POST /v1/audio/speech` — 语音合成（OpenAI 格式）

## 许可证

本项目代码遵循 MIT License。各模型权重文件遵循其各自的许可证。
