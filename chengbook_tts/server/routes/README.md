# routes/ — API 路由

FastAPI 路由模块，按功能拆分。

## 文件说明

| 文件 | 说明 |
|------|------|
| `native.py` | 原生 API — /api/tts, /api/tts/stream, /api/health, /api/voices, /api/emotions |
| `custom_voices.py` | 自定义音色管理 — /api/voices/custom |
| `openai.py` | OpenAI 兼容 API — /v1/audio/speech, /v1/models, /v1/health |
| `profile.py` | 配置文件管理 — /api/profile |

所有路由通过 `Depends(get_engine)` 获取当前引擎，不直接依赖具体模型。
