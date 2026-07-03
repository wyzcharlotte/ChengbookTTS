#!/bin/bash
MODEL_TYPE=${MODEL_TYPE:-cosyvoice3}
PORT=${PORT:-8080}
echo "ChengbookTTS Server — Model: $MODEL_TYPE, Port: $PORT"
python -m chengbook_tts.cli serve --model "$MODEL_TYPE" --port "$PORT"
