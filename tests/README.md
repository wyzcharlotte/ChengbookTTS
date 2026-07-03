# tests/ — 测试套件

ChengbookTTS 的测试代码。

## 文件说明

| 文件/目录 | 说明 |
|-----------|------|
| `conftest.py` | Pytest 共享 fixtures（mock 引擎、测试客户端） |
| `test_engines/test_base.py` | TTSEngine ABC 契约测试 |
| `test_engines/test_cosyvoice3.py` | CosyVoice3 引擎测试 |
| `test_engines/test_soulxpodcast.py` | SoulX-Podcast 引擎测试 |
| `test_engines/test_indextts_v1.py` | IndexTTS v1 引擎测试 |
| `test_engines/test_indextts_v2.py` | IndexTTS v2 引擎测试 |
| `test_server.py` | FastAPI 服务集成测试（使用 TestClient） |
| `test_client.py` | Python SDK 测试（针对模拟服务器） |
| `test_openai_api.py` | OpenAI 兼容 API 测试 |
| `quick_test.py` | 快速冒烟测试（健康检查 + 并发） |

## 运行测试

```bash
# 全部测试
pytest tests/

# 特定引擎测试
pytest tests/test_engines/

# 快速冒烟测试（需要服务运行）
python tests/quick_test.py

# 带覆盖率
pytest tests/ --cov=chengbook_tts
```

## Mock 策略

- 引擎测试使用 **mock 模型**（不加载真实权重），测试接口契约
- 服务测试使用 `TestClient` + mock engine fixture
- 集成测试在 CI 中可选运行（需要 GPU + 模型权重）
