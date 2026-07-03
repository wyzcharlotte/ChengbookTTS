# vendor/ — 模型源码（符号链接）

此目录通过**符号链接（symlink）**指向源项目中的模型代码，
避免代码重复，保持与源项目同步更新。

## 链接说明

| 链接 | 目标 | 说明 |
|------|------|------|
| `cosyvoice` | `../../ChengbookTTS_cosyvoice3_ui/vendor/cosyvoice` | CosyVoice3 模型源码 |
| `matcha` | `../../ChengbookTTS_cosyvoice3_ui/vendor/matcha` | Matcha-TTS 组件 |
| `soulxpodcast` | `../../SoulX-Podcast/soulxpodcast` | SoulX-Podcast 核心包 |
| `indextts` | `../../Index-TTS-Vllm2/index-tts-vllm/indextts` | IndexTTS 核心库 |

## 为什么用符号链接？

1. **避免代码重复**: 不复制源项目的代码
2. **自动同步**: 源项目更新后自动生效
3. **保持独立**: ChengbookTTS 是独立项目，不修改源项目代码

## 如果符号链接不可用

Windows 需要启用开发者模式或管理员权限才能创建符号链接。
如果无法创建，可使用 `sys.path.append()` 作为备选方案（在各 engine 的 `_setup_path()` 中处理）。

## 更新 vendor 代码

直接更新源项目中的代码即可。符号链接会自动反映变化。
如需使用不同版本，可修改符号链接目标或修改 engine 中的 `_setup_path()`。
