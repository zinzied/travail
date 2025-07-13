# 🤖 CV Evaluator - Multi-Language AI Document Chat System

A comprehensive AI-powered system for evaluating CVs and chatting with documents in multiple languages (Arabic, English, French).

**Created by:** [Zied Boughdir](https://github.com/zinzied)  
**GitHub:** [@zinzied](https://github.com/zinzied)

## ✨ Features

### 🌍 **Multi-Language Support**
- **Arabic (العربية)**: Full RTL support with cultural context
- **French (Français)**: Complete business terminology
- **English**: Professional and technical terminology

### 📄 **Universal Document Support**
- **PDF Documents** (.pdf) - Multi-method text extraction
- **Word Documents** (.docx, .doc) - Text and table processing
- **Excel Spreadsheets** (.xlsx, .xls) - Multi-sheet data analysis
- **Text Files** (.txt) - Direct text processing

### 🤖 **25+ AI Models**
- **Premium APIs**: OpenAI GPT-4, Anthropic Claude, Google Gemini
- **Fast Free APIs**: Groq (Llama 3, Mixtral), Cohere
- **Local Models**: Ollama (Llama 3, Mistral, Gemma, Phi-3)
- **Self-Hosted**: LocalAI, LM Studio

### 💬 **Smart AI Chat**
- Context-aware questions based on document type
- Professional analysis in your preferred language
- CV assessment, data analysis, document summarization

## 🚀 Quick Start

### Option 1: Groq API (Fastest, Free)
```bash
# 1. Get free API key: https://console.groq.com
# 2. Set environment variable
set GROQ_API_KEY=your_key_here

# 3. Start application
python run_web_app.py
```

### Option 2: Interactive Setup
```bash
# Run setup wizard
python setup_api.py

# Start application
python run_web_app.py
```

### Option 3: Local Models (Private)
```bash
# 1. Download Ollama: https://ollama.ai
# 2. Install model
ollama pull llama3

# 3. Start server
ollama serve

# 4. Start application (no API key needed)
python run_web_app.py
```

## 📋 Installation

1. **Clone or download** this repository
2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
3. **Configure AI models** (choose one):
   - **Groq API**: Get free key from https://console.groq.com
   - **Ollama**: Download from https://ollama.ai
   - **OpenAI**: Get key from https://platform.openai.com

4. **Start the application**:
   ```bash
   python run_web_app.py
   ```

## 🔧 Configuration

### Quick Configuration
```bash
# Check your current setup
python check_config.py

# Interactive setup wizard
python setup_api.py

# Create .env file manually
python create_env.bat
```

### Manual Configuration
Create `.env` file with your API keys:
```bash
# Fast & Free (Recommended)
GROQ_API_KEY=gsk_your_groq_key_here

# Premium Options
OPENAI_API_KEY=sk-your_openai_key_here
ANTHROPIC_API_KEY=sk-ant-your_anthropic_key_here
GOOGLE_API_KEY=your_google_key_here
```

## 🎯 Usage

1. **Start the app**: `python run_web_app.py`
2. **Select language**: Choose Arabic, French, or English
3. **Configure AI model**: Set up your preferred AI provider
4. **Upload document**: PDF, Word, Excel, or text file
5. **Chat with AI**: Ask questions about your document content

### Example Use Cases
- **HR Professionals**: Analyze CVs in multiple languages
- **Recruiters**: Quick candidate assessment and screening
- **Data Analysts**: Chat with Excel spreadsheets for insights
- **Document Review**: Summarize and analyze Word documents

## 📚 Documentation

- **`API_SETUP_GUIDE.md`**: Detailed API configuration instructions
- **`AI_MODELS_GUIDE.md`**: Complete guide to available AI models
- **`COMPLETE_SETUP_GUIDE.md`**: Comprehensive setup and usage guide

## 🛠️ Troubleshooting

### Common Issues
- **"No models available"**: Run `python check_config.py` to verify setup
- **PDF extraction fails**: Try different PDF or check troubleshooting tips in app
- **API connection errors**: Verify API keys and internet connection

### Getting Help
1. Run configuration checker: `python check_config.py`
2. Check documentation files
3. Verify API keys are correctly set

## 📊 System Requirements

- **Python**: 3.8 or higher
- **Memory**: 4GB RAM minimum (8GB recommended for local models)
- **Storage**: 2GB free space
- **Internet**: Required for API-based models (optional for local models)

## 🔒 Privacy & Security

- **Local Models**: Complete privacy with Ollama (no data sent externally)
- **API Models**: Data sent to respective providers (OpenAI, Groq, etc.)
- **API Keys**: Stored locally in .env file (never committed to version control)

## 🎉 Ready to Use!

Your system supports:
- ✅ **3 Languages** with full UI translation
- ✅ **25+ AI Models** from free to premium
- ✅ **4 Document Types** with smart analysis
- ✅ **Easy Setup** with multiple configuration options

Start chatting with your documents in Arabic, French, or English! 🚀

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.
