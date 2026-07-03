@echo off
echo ========================================
echo   ChengbookTTS — 安装所有依赖
echo ========================================
echo.

echo [1/3] 安装基础依赖...
pip install -r requirements.txt
if %ERRORLEVEL% NEQ 0 (
    echo 基础依赖安装失败！
    pause
    exit /b 1
)

echo.
echo [2/3] 安装 CosyVoice3 依赖...
pip install onnxruntime transformers einops tiktoken whisper
if %ERRORLEVEL% NEQ 0 (
    echo CosyVoice3 依赖安装失败（非致命）！
)

echo.
echo [3/3] 安装客户端依赖（可选）...
pip install sounddevice
if %ERRORLEVEL% NEQ 0 (
    echo 客户端依赖安装失败（非致命）！
)

echo.
echo ========================================
echo   安装完成！
echo ========================================
echo.
echo 启动服务:
echo   python -m chengbook_tts.cli serve
echo.
pause
