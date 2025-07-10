"""
Free AI models integration for CV evaluation and chat.

Created by: Zied Boughdir (@zinzied)
GitHub: https://github.com/zinzied/cv-evaluation-system
"""

import logging
import json
import requests
from typing import Dict, List, Any, Optional
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class FreeAIModel(ABC):
    """Abstract base class for free AI models."""
    
    @abstractmethod
    def generate_response(self, prompt: str, max_tokens: int = 500) -> str:
        """Generate response from the model."""
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """Check if the model is available."""
        pass


class OllamaModel(FreeAIModel):
    """Ollama local AI model integration."""
    
    def __init__(self, model_name: str = "llama2", base_url: str = "http://localhost:11434"):
        self.model_name = model_name
        self.base_url = base_url
        self.api_url = f"{base_url}/api/generate"
    
    def generate_response(self, prompt: str, max_tokens: int = 500) -> str:
        """Generate response using Ollama."""
        try:
            payload = {
                "model": self.model_name,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "num_predict": max_tokens,
                    "temperature": 0.7
                }
            }
            
            response = requests.post(self.api_url, json=payload, timeout=30)
            response.raise_for_status()
            
            result = response.json()
            return result.get("response", "").strip()
            
        except Exception as e:
            logger.error(f"Ollama API error: {e}")
            return f"Error: Could not connect to Ollama. Make sure Ollama is running with model '{self.model_name}'"
    
    def is_available(self) -> bool:
        """Check if Ollama is running."""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            return response.status_code == 200
        except:
            return False


class HuggingFaceModel(FreeAIModel):
    """Hugging Face Transformers model (local)."""
    
    def __init__(self, model_name: str = "microsoft/DialoGPT-medium"):
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """Load the Hugging Face model."""
        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM
            import torch

            # Check if model is already cached locally
            from transformers.utils import cached_file
            try:
                # Try to find cached model without downloading
                cached_file(self.model_name, "config.json", local_files_only=True)

                self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, local_files_only=True)
                self.model = AutoModelForCausalLM.from_pretrained(self.model_name, local_files_only=True)

                # Add padding token if not present
                if self.tokenizer.pad_token is None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token

                logger.info(f"Loaded Hugging Face model: {self.model_name}")

            except Exception:
                # Model not cached locally, don't auto-download
                logger.info(f"Hugging Face model {self.model_name} not available locally. Use 'pip install transformers torch' and download manually if needed.")

        except ImportError:
            logger.info("Transformers library not installed. Install with: pip install transformers torch")
        except Exception as e:
            logger.error(f"Failed to load Hugging Face model: {e}")
    
    def generate_response(self, prompt: str, max_tokens: int = 500) -> str:
        """Generate response using Hugging Face model."""
        if not self.model or not self.tokenizer:
            return "Error: Hugging Face model not loaded. Install transformers: pip install transformers torch"
        
        try:
            import torch
            
            # Encode input
            inputs = self.tokenizer.encode(prompt, return_tensors="pt", max_length=512, truncation=True)
            
            # Generate response
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs,
                    max_length=inputs.shape[1] + max_tokens,
                    num_return_sequences=1,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            # Decode response
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Remove the input prompt from response
            if response.startswith(prompt):
                response = response[len(prompt):].strip()
            
            return response or "I understand your question, but I need more context to provide a helpful response."
            
        except Exception as e:
            logger.error(f"Hugging Face generation error: {e}")
            return f"Error generating response: {e}"
    
    def is_available(self) -> bool:
        """Check if model is loaded."""
        return self.model is not None and self.tokenizer is not None


class OpenAICompatibleModel(FreeAIModel):
    """OpenAI-compatible API (like LocalAI, text-generation-webui)."""
    
    def __init__(self, base_url: str = "http://localhost:5000", model_name: str = "gpt-3.5-turbo"):
        self.base_url = base_url.rstrip('/')
        self.model_name = model_name
        self.api_url = f"{base_url}/v1/chat/completions"
    
    def generate_response(self, prompt: str, max_tokens: int = 500) -> str:
        """Generate response using OpenAI-compatible API."""
        try:
            payload = {
                "model": self.model_name,
                "messages": [
                    {"role": "user", "content": prompt}
                ],
                "max_tokens": max_tokens,
                "temperature": 0.7
            }
            
            response = requests.post(self.api_url, json=payload, timeout=30)
            response.raise_for_status()
            
            result = response.json()
            return result["choices"][0]["message"]["content"].strip()
            
        except Exception as e:
            logger.error(f"OpenAI-compatible API error: {e}")
            return f"Error: Could not connect to API at {self.base_url}"
    
    def is_available(self) -> bool:
        """Check if API is available."""
        try:
            # Try a simple request to check availability
            response = requests.get(f"{self.base_url}/v1/models", timeout=5)
            return response.status_code == 200
        except:
            return False


class FreeModelManager:
    """Manager for free AI models."""
    
    def __init__(self):
        self.models = {}
        self.current_model = None
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize available free models."""
        # Ollama models
        self.models["ollama_llama2"] = OllamaModel("llama2")
        self.models["ollama_codellama"] = OllamaModel("codellama")
        self.models["ollama_mistral"] = OllamaModel("mistral")
        
        # Hugging Face models
        self.models["hf_dialogpt"] = HuggingFaceModel("microsoft/DialoGPT-medium")
        self.models["hf_gpt2"] = HuggingFaceModel("gpt2")
        
        # OpenAI-compatible APIs
        self.models["localai"] = OpenAICompatibleModel("http://localhost:8080")
        self.models["textgen_webui"] = OpenAICompatibleModel("http://localhost:5000")
    
    def get_available_models(self) -> Dict[str, bool]:
        """Get list of available models."""
        available = {}
        for name, model in self.models.items():
            available[name] = model.is_available()
        return available
    
    def set_model(self, model_name: str) -> bool:
        """Set the current model."""
        if model_name in self.models:
            if self.models[model_name].is_available():
                self.current_model = self.models[model_name]
                logger.info(f"Set current model to: {model_name}")
                return True
            else:
                logger.warning(f"Model {model_name} is not available")
                return False
        else:
            logger.error(f"Unknown model: {model_name}")
            return False
    
    def auto_select_model(self) -> Optional[str]:
        """Automatically select the first available model."""
        available = self.get_available_models()
        
        # Priority order for model selection
        priority_order = [
            "ollama_llama2", "ollama_mistral", "ollama_codellama",
            "localai", "textgen_webui",
            "hf_dialogpt", "hf_gpt2"
        ]
        
        for model_name in priority_order:
            if available.get(model_name, False):
                if self.set_model(model_name):
                    return model_name
        
        return None
    
    def generate_response(self, prompt: str, max_tokens: int = 500) -> str:
        """Generate response using current model."""
        if not self.current_model:
            # Try to auto-select a model
            selected = self.auto_select_model()
            if not selected:
                return self._get_fallback_response(prompt)
        
        return self.current_model.generate_response(prompt, max_tokens)
    
    def _get_fallback_response(self, prompt: str) -> str:
        """Provide fallback response when no AI model is available."""
        if "evaluate" in prompt.lower() or "score" in prompt.lower():
            return """I'd be happy to help evaluate this CV, but no AI models are currently available. 

To use free AI models, you can:

1. **Install Ollama** (Recommended):
   - Download from: https://ollama.ai
   - Run: ollama pull llama2
   - Start: ollama serve

2. **Use Hugging Face Transformers**:
   - Install: pip install transformers torch
   - Models will download automatically

3. **Set up LocalAI**:
   - Follow setup at: https://localai.io

The system will work with basic rule-based evaluation without AI models."""

        elif "chat" in prompt.lower() or "help" in prompt.lower():
            return """Hello! I'm the CV Evaluator assistant. While I don't have access to AI models right now, I can still help you with:

• CV evaluation using rule-based scoring
• Batch processing of multiple CVs
• Creating custom evaluation criteria
• Generating professional reports

To enable AI-powered features, please set up one of these free options:
- Ollama (easiest): https://ollama.ai
- Hugging Face Transformers
- LocalAI or compatible APIs

How can I assist you today?"""

        else:
            return """I understand you're asking about CV evaluation. While AI models aren't currently available, the system can still:

• Extract and parse CV content
• Score based on criteria matching
• Generate detailed reports
• Process multiple files

For AI-enhanced evaluation, please set up a free model like Ollama or Hugging Face Transformers."""


# Global model manager instance
model_manager = FreeModelManager()


def get_ai_response(prompt: str, max_tokens: int = 500) -> str:
    """Get AI response using available free models."""
    return model_manager.generate_response(prompt, max_tokens)


def list_available_models() -> Dict[str, bool]:
    """List all available AI models."""
    return model_manager.get_available_models()


def set_ai_model(model_name: str) -> bool:
    """Set the AI model to use."""
    return model_manager.set_model(model_name)


def auto_select_ai_model() -> Optional[str]:
    """Auto-select the best available AI model."""
    return model_manager.auto_select_model()
