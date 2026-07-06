---
name: chengbooktts-project-overview
description: ChengbookTTS v2.0.0 — unified multi-model TTS platform overview
metadata:
  type: project
---

ChengbookTTS is a unified multi-model TTS platform (v2.0.0) at `e:/program/ChengbookTTS/`. It wraps CosyVoice3, SoulX-Podcast 1.7B, IndexTTS v1/v1.5, and IndexTTS-2 behind a common TTSEngine ABC.

Architecture: engines/ (base → 4 engine implementations) → ModelManager (singleton, one model on GPU at a time) → FastAPI server (routes/, console.py, concurrency.py) + CLI (cli/main.py) + Python SDK (client/sdk.py).

Key design: strategy pattern with ABC, pre-cached voice embeddings on startup, capability flags per engine for UI adaptation, 8 universal emotions mapped differently per model, OpenAI-compatible /v1/audio/speech endpoint.

**Why:** This is the main project being worked on. Understanding the full architecture is essential for any modifications.
**How to apply:** Read `项目分析总结.md` for the comprehensive overview. Key files: `chengbook_tts/engines/base.py`, `chengbook_tts/engines/manager.py`, `chengbook_tts/server/app.py`, `chengbook_tts/server/routes/native.py`.
