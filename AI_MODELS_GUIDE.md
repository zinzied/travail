# 🤖 AI Models Guide for Document Chat

## Overview

The CV Evaluator system supports multiple AI models for document analysis and chat functionality. This guide covers the available options, setup instructions, and recommendations for different use cases.

## 🚀 Recommended AI Models

### 1. Groq API (Fastest & Free)
**Best for: Production use, fast responses, multi-language support**

- **Models Available**: Llama 3, Mixtral 8x7B
- **Speed**: Ultra-fast (< 1 second response)
- **Cost**: Free tier with generous limits
- **Languages**: Excellent support for English, French, Arabic
- **Setup**:
  1. Get free API key: https://console.groq.com
  2. Set environment variable: `GROQ_API_KEY=your_key_here`
  3. Restart the application

### 2. Ollama (Local & Private)
**Best for: Privacy, offline use, customization**

- **Models Available**: Llama 3, Mistral, Gemma, Phi-3, Qwen
- **Speed**: Fast (2-5 seconds response)
- **Cost**: Completely free
- **Languages**: Good support for multiple languages
- **Setup**:
  1. Download Ollama: https://ollama.ai
  2. Install a model: `ollama pull llama3`
  3. Start server: `ollama serve`

### 3. Hugging Face Transformers (Offline)
**Best for: Offline use, research, experimentation**

- **Models Available**: FLAN-T5, DialoGPT, GPT-2
- **Speed**: Slower (5-15 seconds)
- **Cost**: Free
- **Languages**: Variable support
- **Setup**: Models download automatically

## 📋 Detailed Model Comparison

| Model | Speed | Quality | Languages | Privacy | Setup Difficulty |
|-------|-------|---------|-----------|---------|------------------|
| Groq Llama3 | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| Ollama Llama3 | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| Ollama Mistral | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| HF FLAN-T5 | ⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |

## 🌍 Multi-Language Support

### Arabic Language Support
- **Best Models**: Groq Llama3, Ollama Llama3, Ollama Qwen
- **Features**: 
  - Right-to-left text handling
  - Arabic document analysis
  - Cultural context understanding

### French Language Support
- **Best Models**: Groq Llama3, Ollama Llama3, Ollama Mistral
- **Features**:
  - French document analysis
  - Professional terminology
  - European context understanding

### English Language Support
- **Best Models**: All models support English well
- **Features**:
  - Comprehensive analysis
  - Professional insights
  - Technical terminology

## 🛠️ Setup Instructions

### Groq API Setup (Recommended)

1. **Create Account**:
   - Visit: https://console.groq.com
   - Sign up for free account
   - Navigate to API Keys section

2. **Get API Key**:
   - Click "Create API Key"
   - Copy the generated key
   - Keep it secure

3. **Set Environment Variable**:
   ```bash
   # Windows
   set GROQ_API_KEY=your_key_here
   
   # Linux/Mac
   export GROQ_API_KEY=your_key_here
   
   # Or create .env file
   echo "GROQ_API_KEY=your_key_here" > .env
   ```

4. **Restart Application**:
   - Close the web app
   - Run `python run_web_app.py` again

### Ollama Setup (Local)

1. **Download Ollama**:
   - Visit: https://ollama.ai
   - Download for your OS
   - Install the application

2. **Install Models**:
   ```bash
   # Recommended models
   ollama pull llama3        # Best overall
   ollama pull mistral       # Good for French
   ollama pull qwen          # Good for multilingual
   ollama pull gemma         # Fast and efficient
   ```

3. **Start Server**:
   ```bash
   ollama serve
   ```

4. **Verify Installation**:
   - Open web app
   - Check AI Model Settings
   - Look for green checkmarks

### Hugging Face Setup

1. **Install Dependencies**:
   ```bash
   pip install transformers torch
   ```

2. **Models Download Automatically**:
   - First use will download models
   - Requires internet connection initially
   - Models cached locally after download

## 🎯 Use Case Recommendations

### For HR Professionals
- **Primary**: Groq Llama3 (fast, professional responses)
- **Backup**: Ollama Llama3 (privacy for sensitive documents)

### For Recruiters
- **Primary**: Groq Llama3 (quick candidate assessment)
- **Secondary**: Ollama Mistral (good for French candidates)

### For Personal Use
- **Primary**: Ollama Llama3 (free, private, no API keys)
- **Secondary**: Hugging Face models (completely offline)

### For Multilingual Organizations
- **Arabic**: Groq Llama3 or Ollama Qwen
- **French**: Groq Llama3 or Ollama Mistral
- **English**: Any model works well

## 🔧 Troubleshooting

### Groq API Issues
- **"API key not found"**: Set GROQ_API_KEY environment variable
- **"Rate limit exceeded"**: Wait a few minutes or upgrade plan
- **"Connection error"**: Check internet connection

### Ollama Issues
- **"Model not available"**: Run `ollama pull model_name`
- **"Connection refused"**: Start Ollama server with `ollama serve`
- **"Slow responses"**: Try smaller models like `phi3`

### General Issues
- **No models available**: Check installation and restart app
- **Poor responses**: Try different model or check language settings
- **Memory issues**: Use smaller models or increase system RAM

## 📊 Performance Tips

### For Best Speed
1. Use Groq API models
2. Keep questions concise
3. Use stable internet connection

### For Best Quality
1. Use Llama3-based models
2. Provide clear, specific questions
3. Select appropriate language setting

### For Privacy
1. Use Ollama local models
2. Avoid cloud-based APIs
3. Keep documents on local machine

## 🔮 Future Models

We're continuously adding support for new models:
- **Anthropic Claude** (API integration planned)
- **Google Gemini** (API integration planned)
- **Local LLaMA variants** (More Ollama models)
- **Specialized models** (Document-specific fine-tuned models)

---

*This guide is updated regularly. Check back for new model additions and improved setup instructions.*
