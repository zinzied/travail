# 🔑 API Configuration Guide

## Overview

This guide shows you how to configure various AI model APIs for the CV Evaluator system. You can choose from multiple providers based on your needs for speed, cost, privacy, and features.

## 🚀 Quick Setup (Recommended)

### Option 1: Groq API (Fastest & Free)
```bash
# 1. Get free API key from: https://console.groq.com
# 2. Copy .env.template to .env
cp .env.template .env

# 3. Edit .env file and add your key:
GROQ_API_KEY=your_groq_api_key_here

# 4. Restart the application
python run_web_app.py
```

### Option 2: Ollama (Local & Private)
```bash
# 1. Download and install Ollama: https://ollama.ai
# 2. Install a model
ollama pull llama3

# 3. Start Ollama server
ollama serve

# 4. No API key needed - restart the application
python run_web_app.py
```

## 📋 All Available APIs

### 1. **Groq API** (Recommended - Fast & Free)
- **Website**: https://console.groq.com
- **Models**: Llama 3, Mixtral, Gemma
- **Speed**: Ultra-fast (< 1 second)
- **Cost**: Free with generous limits
- **Setup**:
  ```bash
  # Get API key from console.groq.com
  export GROQ_API_KEY=your_key_here
  # Or add to .env file:
  echo "GROQ_API_KEY=your_key_here" >> .env
  ```

### 2. **OpenAI API** (Premium Quality)
- **Website**: https://platform.openai.com
- **Models**: GPT-4, GPT-3.5-turbo
- **Speed**: Fast
- **Cost**: Pay per use ($0.01-$0.06 per 1K tokens)
- **Setup**:
  ```bash
  # Get API key from platform.openai.com
  export OPENAI_API_KEY=your_key_here
  # Or add to .env file:
  echo "OPENAI_API_KEY=your_key_here" >> .env
  ```

### 3. **Anthropic Claude API**
- **Website**: https://console.anthropic.com
- **Models**: Claude 3 (Opus, Sonnet, Haiku)
- **Speed**: Fast
- **Cost**: Pay per use ($0.015-$0.075 per 1K tokens)
- **Setup**:
  ```bash
  # Get API key from console.anthropic.com
  export ANTHROPIC_API_KEY=your_key_here
  # Or add to .env file:
  echo "ANTHROPIC_API_KEY=your_key_here" >> .env
  ```

### 4. **Google Gemini API**
- **Website**: https://makersuite.google.com
- **Models**: Gemini Pro
- **Speed**: Fast
- **Cost**: Free tier available
- **Setup**:
  ```bash
  # Get API key from makersuite.google.com
  export GOOGLE_API_KEY=your_key_here
  # Or add to .env file:
  echo "GOOGLE_API_KEY=your_key_here" >> .env
  ```

### 5. **Ollama** (Local & Private)
- **Website**: https://ollama.ai
- **Models**: Llama 3, Mistral, Gemma, Phi-3, Qwen
- **Speed**: Fast (local processing)
- **Cost**: Free (uses your hardware)
- **Setup**:
  ```bash
  # Download and install Ollama
  # Then install models:
  ollama pull llama3      # Best overall
  ollama pull mistral     # Good for French
  ollama pull qwen        # Good for multilingual
  
  # Start server:
  ollama serve
  ```

### 6. **Hugging Face API**
- **Website**: https://huggingface.co
- **Models**: Llama 2, Mistral, CodeLlama
- **Speed**: Medium
- **Cost**: Free tier available
- **Setup**:
  ```bash
  # Get API token from huggingface.co/settings/tokens
  export HUGGINGFACE_API_KEY=your_token_here
  # Or add to .env file:
  echo "HUGGINGFACE_API_KEY=your_token_here" >> .env
  ```

## 🔧 Configuration Methods

### Method 1: Environment Variables
```bash
# Set directly in terminal (temporary)
export GROQ_API_KEY=your_key_here
export OPENAI_API_KEY=your_key_here

# Run application
python run_web_app.py
```

### Method 2: .env File (Recommended)
```bash
# 1. Copy template
cp .env.template .env

# 2. Edit .env file with your favorite editor
nano .env
# or
code .env

# 3. Add your API keys:
GROQ_API_KEY=your_groq_key_here
OPENAI_API_KEY=your_openai_key_here
ANTHROPIC_API_KEY=your_anthropic_key_here

# 4. Save and restart application
python run_web_app.py
```

### Method 3: System Environment (Permanent)
```bash
# Linux/Mac - Add to ~/.bashrc or ~/.zshrc
echo 'export GROQ_API_KEY=your_key_here' >> ~/.bashrc
source ~/.bashrc

# Windows - Set system environment variable
setx GROQ_API_KEY "your_key_here"
```

## 🧪 Testing Your Configuration

### In the Web Interface:
1. Open the application: `python run_web_app.py`
2. Go to "AI Chat" mode
3. Expand "🤖 AI Model Settings"
4. Check the "🔑 API Status" section
5. Use "Test Connection" buttons

### Command Line Test:
```bash
# Test Groq API
python -c "
from cv_evaluator.utils.api_config import api_config
print('Groq enabled:', api_config.is_provider_enabled('groq'))
"

# Test all providers
python -c "
from cv_evaluator.utils.api_config import api_config
for provider in ['groq', 'openai', 'anthropic', 'google', 'ollama']:
    print(f'{provider}: {api_config.is_provider_enabled(provider)}')
"
```

## 💡 Recommendations by Use Case

### For HR Professionals (Production Use)
1. **Primary**: Groq API (fast, free, reliable)
2. **Backup**: OpenAI GPT-3.5-turbo (premium quality)
3. **Local**: Ollama Llama3 (for sensitive documents)

### For Personal Use
1. **Primary**: Ollama (completely free and private)
2. **Secondary**: Groq API (for faster responses)

### For Developers/Testing
1. **Primary**: Groq API (fast iteration)
2. **Secondary**: Multiple APIs for comparison

### For Enterprise/Sensitive Data
1. **Primary**: Ollama (on-premises)
2. **Secondary**: LocalAI or LM Studio (self-hosted)

## 🔒 Security Best Practices

### API Key Security:
- **Never commit API keys to version control**
- **Use .env files and add .env to .gitignore**
- **Rotate API keys regularly**
- **Use environment variables in production**

### Example .gitignore:
```
.env
*.env
.env.local
.env.production
```

## 🚨 Troubleshooting

### Common Issues:

#### "No API key found"
- Check if API key is set in environment or .env file
- Restart the application after setting keys
- Verify key format (no extra spaces or quotes)

#### "Connection failed"
- Check internet connection for API-based models
- Verify API key is valid and not expired
- Check API provider status page

#### "Rate limit exceeded"
- Wait a few minutes before retrying
- Consider upgrading to paid tier
- Switch to different provider temporarily

#### Ollama "Connection refused"
- Make sure Ollama is installed and running
- Start Ollama server: `ollama serve`
- Check if model is installed: `ollama list`

### Getting Help:
1. Check provider documentation
2. Test with simple API calls
3. Verify network connectivity
4. Check application logs

## 📊 Cost Comparison

| Provider | Free Tier | Paid Pricing | Speed | Quality |
|----------|-----------|--------------|-------|---------|
| Groq | Generous free | $0.27/1M tokens | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| OpenAI | $5 credit | $0.50-$30/1M tokens | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| Anthropic | $5 credit | $15-$75/1M tokens | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| Google | Free tier | $0.50-$7/1M tokens | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| Ollama | Free | Hardware cost | ⭐⭐⭐ | ⭐⭐⭐⭐ |

## 🎯 Next Steps

1. **Choose your provider** based on needs
2. **Get API keys** from provider websites
3. **Configure using .env file** method
4. **Test connection** in web interface
5. **Start using** the AI chat features!

---

*For more help, see the main documentation or create an issue on GitHub.*
