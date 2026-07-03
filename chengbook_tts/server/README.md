# server/ — FastAPI 服务

基于 FastAPI 的 HTTP 服务，提供原生 API + OpenAI 兼容 API + Web 控制台。

## 文件说明

| 文件 | 说明 |
|------|------|
| `app.py` | FastAPI 应用工厂 — 创建 app、配置 lifespan（模型加载/卸载） |
| `dependencies.py` | FastAPI `Depends()` 函数 — `get_engine()`、`get_model_manager()` |
| `concurrency.py` | 并发控制 — asyncio.Semaphore + ThreadPoolExecutor |
| `models.py` | Pydantic 请求/响应模型 |
| `audio_utils.py` | 音频格式转换 — WAV 封装、ffmpeg 转码 |
| `console.py` | Web 控制台 HTML（内联 SPA 页面） |
| `routes/native.py` | 原生 API — `/api/tts`, `/api/tts/stream`, `/api/health`, `/api/voices`, `/api/emotions` |
| `routes/custom_voices.py` | 自定义音色 API — `/api/voices/custom` |
| `routes/openai.py` | OpenAI 兼容 API — `/v1/audio/speech`, `/v1/models`, `/v1/health` |
| `routes/profile.py` | 配置文件 API — `/api/profile` |

## API 端点

### 原生 API (`/api/*`)
| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/api/health` | 引擎状态检查 |
| GET | `/api/voices` | 可用音色列表 |
| GET | `/api/emotions` | 可用情绪列表 |
| POST | `/api/tts` | 非流式语音合成（返回 WAV） |
| POST | `/api/tts/stream` | 流式语音合成（SSE） |
| POST | `/api/voices/custom` | 上传自定义音色 |
| DELETE | `/api/voices/custom/{voice_id}` | 删除自定义音色 |
| GET/POST | `/api/profile` | 读取/更新配置文件 |

### OpenAI 兼容 API (`/v1/*`)
| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/v1/models` | 模型列表 |
| GET | `/v1/health` | 健康检查 |
| POST | `/v1/audio/speech` | 语音合成（OpenAI 格式） |

## 设计原则

- **模型无关**: 所有 route handler 通过 `Depends(get_engine)` 获取引擎，不直接依赖具体模型
- **线程安全**: GPU 推理通过 `ThreadPoolExecutor` 串行执行
- **并发限制**: `asyncio.Semaphore` 限制同时进行的推理数
