# profiles/ — 运行时配置文件

存储用户运行时的偏好设置：当前音色、情绪、语速、分段偏好。

## 文件说明

| 文件 | 说明 |
|------|------|
| `current.json` | 当前活跃的配置文件 |

## 配置文件格式

```json
{
  "voice": "woman",
  "emotion": "calm",
  "speed": 1.0,
  "segment": true,
  "model_type": "cosyvoice3"
}
```

## 使用方式

- **Web 控制台**: 在 / 页面上可视化切换，自动保存
- **API**: GET /api/profile 读取，POST /api/profile 更新
- **OpenAI 兼容 API**: /v1/audio/speech 会读取此配置文件来确定默认音色和情绪
