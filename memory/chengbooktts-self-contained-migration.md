---
name: chengbooktts-self-contained-migration
description: Project made self-contained — all external model/code dependencies copied into project
metadata:
  type: project
---

On 2026-07-06, the project was made fully self-contained. Previously it relied on sibling directories (`../ChengbookTTS_cosyvoice3_ui/`, `../SoulX-Podcast/`, `../Index-TTS-Vllm2/`) via relative paths.

Changes made:
1. `vendor/cosyvoice/` (2.7MB) — replaced symlink with actual copy from `ChengbookTTS_cosyvoice3_ui/vendor/`
2. `vendor/matcha/` (513KB) — replaced symlink with actual copy
3. `vendor/soulxpodcast/` (418KB) — copied from `../SoulX-Podcast/soulxpodcast/`
4. `vendor/indextts/` (26MB) — copied from `../Index-TTS-Vllm2/index-tts-vllm/indextts/`
5. `models/SoulX-Podcast-1.7B/` (5.4GB) — copied model weights
6. `models/Index-TTS-1.5-vLLM/` (2.8GB) — copied model weights
7. `models/IndexTTS-2-vLLM/` (12GB) — copied model weights
8. `prompts/soulxpodcast/` — copied example audios for voice cloning

Code path updates:
- `engines/manager.py`: `_auto_detect_model_dir()` — all paths now resolve to `project_root/models/`
- `engines/soulxpodcast.py`: `_setup_path()` → vendor/; wav fallback → prompts/soulxpodcast/
- `engines/indextts_v1.py`: `_setup_path()` → vendor/
- `engines/indextts_v2.py`: `_setup_path()` → vendor/
- `config/model_configs/soulxpodcast.yaml`: updated comments

.gitignore excludes: vendor/{cosyvoice,matcha,soulxpodcast,indextts}/, prompts/soulxpodcast/, models/, *.pt, *.safetensors, *.onnx

**Why:** The project needed to be standalone without dependencies on sibling directories. All external source and model files are now self-contained.
**How to apply:** When adding new models or updating paths, always reference project-local paths (models/ for weights, vendor/ for source, prompts/ for audio).

---

On 2026-07-06 (session 2), WebUI prompt_text input was conditionally disabled:
- Only SoulXPodcast actually uses `prompt_text` in `register_voice()` and synthesis
- CosyVoice3's `register_voice()` accepts but ignores `prompt_text` (only extracts audio features)
- IndexTTS v1/v2 also don't use it
- **Solution**: Gray out (disable) the input when `activeModel !== 'soulxpodcast'`, with a red warning note and updated placeholder. Input re-enables automatically on model switch.
- **Why gray out vs hide**: Consistent with existing `section-disabled` pattern; users can discover the feature exists; layout stays stable.
- **Files changed**: `chengbook_tts/server/console.py` (CSS + HTML + `updateCapUI()` + `uploadVoice()`)
