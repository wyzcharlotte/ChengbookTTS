@echo off
echo ========================================
echo   ChengbookTTS Server
echo ========================================
echo.

REM 默认模型和端口
set MODEL_TYPE=%MODEL_TYPE%
if "%MODEL_TYPE%"=="" set MODEL_TYPE=cosyvoice3
set PORT=%PORT%
if "%PORT%"=="" set PORT=8080

echo Model: %MODEL_TYPE%
echo Port: %PORT%
echo.

python -m chengbook_tts.cli serve --model %MODEL_TYPE% --port %PORT%

pause
