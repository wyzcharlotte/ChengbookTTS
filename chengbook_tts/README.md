# chengbook_tts 主包

ChengbookTTS 统一 TTS 平台的核心 Python 包。

## 模块概览

| 模块 | 说明 |
|------|------|
| `config/` | 全局配置中心 — 设置、音色、情绪预设、模型 YAML 配置 |
| `engines/` | TTS 引擎抽象层 — ABC 基类 + 模型适配器 + 模型管理器 |
| `server/` | FastAPI 服务 — 原生 API + OpenAI 兼容 API + Web 控制台 |
| `cli/` | 命令行工具 — 模型切换、启动服务、直接合成、音色管理 |
| `client/` | Python SDK — 同步/流式客户端、交互式测试客户端 |
| `vendor/` | 模型源码（符号链接指向源项目） |
| `prompts/` | 音色提示音频（符号链接 + 自定义上传） |
| `profiles/` | 运行时配置文件（音色/情绪/语速偏好） |
| `utils/` | 工具函数 — 音频处理、文本预处理、日志 |
| `tools/` | 辅助脚本 — 音频格式转换等 |
