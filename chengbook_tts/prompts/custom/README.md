# prompts/custom/ — 自定义音色

用户通过 API 上传的自定义音色 WAV 文件存放于此。

## 文件命名规则

```
custom_{voice_id}.wav
```

例如:
- `custom_5d3f71d6.wav` — voice_id 为 `5d3f71d6` 的音色
- `custom_my_voice.wav` — voice_id 为 `my_voice` 的音色

## 管理方式

通过 API 管理自定义音色：
- `POST /api/voices/custom` — 上传新音色
- `DELETE /api/voices/custom/{voice_id}` — 删除音色

元数据持久化在项目根目录的 `custom_voices.json` 中。
