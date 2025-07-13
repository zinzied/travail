@echo off
echo 🔑 Creating .env file for API configuration
echo ============================================
echo.

REM Check if .env already exists
if exist .env (
    echo ⚠️  .env file already exists!
    echo Current content:
    echo ----------------
    type .env
    echo ----------------
    echo.
    set /p overwrite="Do you want to overwrite it? (y/n): "
    if /i not "%overwrite%"=="y" (
        echo Cancelled. Existing .env file kept.
        pause
        exit /b
    )
)

echo Creating new .env file...
echo.

REM Create .env file with template
(
echo # AI Model API Configuration
echo # Add your API keys below
echo.
echo # Groq API ^(Fast ^& Free^) - Get from: https://console.groq.com
echo GROQ_API_KEY=
echo.
echo # OpenAI API ^(Premium^) - Get from: https://platform.openai.com  
echo OPENAI_API_KEY=
echo.
echo # Anthropic Claude API - Get from: https://console.anthropic.com
echo ANTHROPIC_API_KEY=
echo.
echo # Google Gemini API - Get from: https://makersuite.google.com
echo GOOGLE_API_KEY=
echo.
echo # Hugging Face API - Get from: https://huggingface.co/settings/tokens
echo HUGGINGFACE_API_KEY=
echo.
echo # Local model settings ^(no API key needed^)
echo OLLAMA_BASE_URL=http://localhost:11434
echo.
echo # Application settings
echo DEFAULT_LANGUAGE=en
echo MAX_TOKENS=2000
echo AI_TEMPERATURE=0.7
) > .env

echo ✅ .env file created successfully!
echo.
echo 📝 Next steps:
echo 1. Edit .env file and add your API keys
echo 2. Get Groq API key from: https://console.groq.com ^(recommended^)
echo 3. Save the file
echo 4. Run: python run_web_app.py
echo.
echo 🔧 To edit .env file now, run: notepad .env
echo.

set /p edit="Do you want to edit .env file now? (y/n): "
if /i "%edit%"=="y" (
    notepad .env
)

echo.
echo 🚀 Setup complete! Run 'python run_web_app.py' to start the application.
pause
