#!/bin/bash
echo "ChengbookTTS — Installing all dependencies..."
pip install -r requirements.txt
pip install onnxruntime transformers einops tiktoken whisper  # CosyVoice3
pip install sounddevice  # client
echo "Done. Run: python -m chengbook_tts.cli serve"
