# utils/ — 工具函数

项目中复用的工具函数。

## 文件说明

| 文件 | 说明 |
|------|------|
| `audio.py` | 音频处理 — 归一化、静音裁剪、WAV 读写、重采样 |
| `text.py` | 中文文本预处理 — jieba 分词 + 歧义词 `[]` 包裹 |
| `logging.py` | 日志配置 — 统一格式和级别控制 |

## 使用方式

```python
from chengbook_tts.utils import (
    normalize_audio,      # 峰值归一化
    trim_silence,         # 裁剪首尾静音
    save_wav,             # 保存 WAV 文件
    load_wav,             # 加载 WAV 文件
    preprocess_chinese_text,  # 中文文本预处理
    setup_logging,        # 配置日志
)
```

## 文本预处理说明

`preprocess_chinese_text()` 使用 jieba 对中文文本进行分词，
将相邻的二字词合并为词组并用 `[ ]` 包裹，减少 LLM 断句歧义。

```
输入: "我的发票抬头是个人，可以开个人发票吗"
输出: "我的 [发票抬头] 是 [个人] ， [可以] 开 [个人发票] 吗"
```

可通过环境变量控制：
- `TEXT_PREPROCESS=0` 完全禁用
- `TEXT_SEGMENT=0` 禁用 `[]` 包裹但保留分词
