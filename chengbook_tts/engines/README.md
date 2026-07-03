# engines/ — TTS 引擎抽象层

所有 TTS 模型通过统一的 `TTSEngine` 抽象基类接入，服务代码无需感知底层模型差异。

## 文件说明

| 文件 | 说明 |
|------|------|
| `base.py` | `TTSEngine` ABC — 定义所有引擎必须实现的接口 |
| `manager.py` | `ModelManager` — 模型生命周期管理（懒加载、切换、卸载） |
| `factory.py` | `create_engine()` — 引擎工厂函数（自动检测模型类型） |
| `cosyvoice3.py` | `CosyVoice3Engine` — CosyVoice3 引擎实现 |
| `soulxpodcast.py` | `SoulXPodcastEngine` — SoulX-Podcast 引擎适配器 |
| `indextts_v1.py` | `IndexTTSEngine` — IndexTTS v1/v1.5 引擎适配器 |
| `indextts_v2.py` | `IndexTTS2Engine` — IndexTTS-2 引擎适配器 |

## TTSEngine ABC 接口

每个引擎必须实现以下核心方法：

```python
class TTSEngine(ABC):
    def initialize(self)          # 加载模型、预计算缓存
    def unload(self)              # 释放 GPU 内存
    def synthesize(self, text, voice, emotion, speed)  # 非流式合成
    def synthesize_stream(self, text, voice, emotion)  # 流式合成
    def register_voice(self, voice_id, wav_path, ...)  # 注册自定义音色
    def unregister_voice(self, voice_id)               # 注销音色
    def health_info(self)         # 引擎状态信息
```

## 模型生命周期

```
启动服务
  └→ ModelManager.load_model(model_type)
       ├→ 卸载旧模型（如有）: engine.unload() → torch.cuda.empty_cache()
       ├→ 懒加载引擎类: importlib.import_module()
       ├→ engine = SomeEngine(voices, emotions, config)
       └→ engine.initialize()  # 加载权重、预热
```

**关键原则**: 同一时刻只有一个模型驻留在 GPU 显存中。

## 添加新引擎

1. 继承 `TTSEngine` ABC
2. 实现所有抽象方法
3. 在 `manager.py` 的 `ENGINE_CLASS_MAP` 中注册
4. 创建对应的 `model_configs/<name>.yaml`
5. 在 `factory.py` 中添加自动检测逻辑（可选）
