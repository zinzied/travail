"""
API Configuration management for AI models.
Handles API keys, endpoints, and model settings.

Created by: Zied Boughdir (@zinzied)
GitHub: https://github.com/zinzied/cv-evaluation-system
"""

import os
import logging
from typing import Dict, Optional, Any
from pathlib import Path
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

class APIConfig:
    """Centralized API configuration management."""
    
    def __init__(self):
        # Load environment variables from .env file
        self._load_env_file()
        self.configs = self._initialize_configs()
    
    def _load_env_file(self):
        """Load environment variables from .env file if it exists."""
        env_file = Path('.env')
        if env_file.exists():
            load_dotenv(env_file)
            logger.info("Loaded configuration from .env file")
        else:
            logger.info("No .env file found, using system environment variables")
    
    def _initialize_configs(self) -> Dict[str, Dict[str, Any]]:
        """Initialize all API configurations."""
        return {
            # Groq API (Fast, Free)
            'groq': {
                'api_key': os.getenv('GROQ_API_KEY'),
                'base_url': 'https://api.groq.com/openai/v1',
                'models': {
                    'llama3-8b': 'llama3-8b-8192',
                    'llama3-70b': 'llama3-70b-8192',
                    'mixtral': 'mixtral-8x7b-32768',
                    'gemma': 'gemma-7b-it'
                },
                'default_model': 'llama3-8b-8192',
                'max_tokens': 4000,
                'temperature': 0.7,
                'enabled': bool(os.getenv('GROQ_API_KEY'))
            },
            
            # OpenAI API (Premium)
            'openai': {
                'api_key': os.getenv('OPENAI_API_KEY'),
                'base_url': 'https://api.openai.com/v1',
                'models': {
                    'gpt-4': 'gpt-4',
                    'gpt-4-turbo': 'gpt-4-turbo-preview',
                    'gpt-3.5-turbo': 'gpt-3.5-turbo',
                    'gpt-3.5-turbo-16k': 'gpt-3.5-turbo-16k'
                },
                'default_model': 'gpt-3.5-turbo',
                'max_tokens': 2000,
                'temperature': 0.7,
                'enabled': bool(os.getenv('OPENAI_API_KEY'))
            },
            
            # Anthropic Claude API
            'anthropic': {
                'api_key': os.getenv('ANTHROPIC_API_KEY'),
                'base_url': 'https://api.anthropic.com',
                'models': {
                    'claude-3-opus': 'claude-3-opus-20240229',
                    'claude-3-sonnet': 'claude-3-sonnet-20240229',
                    'claude-3-haiku': 'claude-3-haiku-20240307'
                },
                'default_model': 'claude-3-sonnet-20240229',
                'max_tokens': 2000,
                'temperature': 0.7,
                'enabled': bool(os.getenv('ANTHROPIC_API_KEY'))
            },
            
            # Google Gemini API
            'google': {
                'api_key': os.getenv('GOOGLE_API_KEY'),
                'base_url': 'https://generativelanguage.googleapis.com/v1beta',
                'models': {
                    'gemini-pro': 'gemini-pro',
                    'gemini-pro-vision': 'gemini-pro-vision'
                },
                'default_model': 'gemini-pro',
                'max_tokens': 2000,
                'temperature': 0.7,
                'enabled': bool(os.getenv('GOOGLE_API_KEY'))
            },
            
            # Cohere API
            'cohere': {
                'api_key': os.getenv('COHERE_API_KEY'),
                'base_url': 'https://api.cohere.ai/v1',
                'models': {
                    'command': 'command',
                    'command-light': 'command-light',
                    'command-nightly': 'command-nightly'
                },
                'default_model': 'command',
                'max_tokens': 2000,
                'temperature': 0.7,
                'enabled': bool(os.getenv('COHERE_API_KEY'))
            },
            
            # Hugging Face API
            'huggingface': {
                'api_key': os.getenv('HUGGINGFACE_API_KEY'),
                'base_url': 'https://api-inference.huggingface.co/models',
                'models': {
                    'llama2-7b': 'meta-llama/Llama-2-7b-chat-hf',
                    'llama2-13b': 'meta-llama/Llama-2-13b-chat-hf',
                    'mistral-7b': 'mistralai/Mistral-7B-Instruct-v0.1',
                    'codellama': 'codellama/CodeLlama-7b-Instruct-hf'
                },
                'default_model': 'meta-llama/Llama-2-7b-chat-hf',
                'max_tokens': 2000,
                'temperature': 0.7,
                'enabled': bool(os.getenv('HUGGINGFACE_API_KEY'))
            },
            
            # Ollama (Local)
            'ollama': {
                'api_key': None,  # No API key needed
                'base_url': os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434'),
                'models': {
                    'llama3': 'llama3',
                    'llama2': 'llama2',
                    'mistral': 'mistral',
                    'codellama': 'codellama',
                    'gemma': 'gemma',
                    'phi3': 'phi3',
                    'qwen': 'qwen'
                },
                'default_model': 'llama3',
                'max_tokens': 2000,
                'temperature': 0.7,
                'enabled': True  # Always enabled if Ollama is running
            },
            
            # LocalAI (Self-hosted)
            'localai': {
                'api_key': os.getenv('LOCALAI_API_KEY'),
                'base_url': os.getenv('LOCALAI_BASE_URL', 'http://localhost:8080'),
                'models': {
                    'gpt-3.5-turbo': 'gpt-3.5-turbo',
                    'gpt-4': 'gpt-4'
                },
                'default_model': 'gpt-3.5-turbo',
                'max_tokens': 2000,
                'temperature': 0.7,
                'enabled': bool(os.getenv('LOCALAI_BASE_URL'))
            },
            
            # LM Studio (Local)
            'lmstudio': {
                'api_key': None,
                'base_url': os.getenv('LMSTUDIO_BASE_URL', 'http://localhost:1234'),
                'models': {
                    'local-model': 'local-model'
                },
                'default_model': 'local-model',
                'max_tokens': 2000,
                'temperature': 0.7,
                'enabled': bool(os.getenv('LMSTUDIO_BASE_URL'))
            }
        }
    
    def get_config(self, provider: str) -> Optional[Dict[str, Any]]:
        """Get configuration for a specific provider."""
        return self.configs.get(provider)
    
    def is_provider_enabled(self, provider: str) -> bool:
        """Check if a provider is enabled and configured."""
        config = self.get_config(provider)
        if not config:
            return False
        return config.get('enabled', False)
    
    def get_enabled_providers(self) -> Dict[str, Dict[str, Any]]:
        """Get all enabled providers."""
        return {
            provider: config 
            for provider, config in self.configs.items() 
            if config.get('enabled', False)
        }
    
    def get_api_key(self, provider: str) -> Optional[str]:
        """Get API key for a provider."""
        config = self.get_config(provider)
        return config.get('api_key') if config else None
    
    def get_base_url(self, provider: str) -> Optional[str]:
        """Get base URL for a provider."""
        config = self.get_config(provider)
        return config.get('base_url') if config else None
    
    def get_models(self, provider: str) -> Dict[str, str]:
        """Get available models for a provider."""
        config = self.get_config(provider)
        return config.get('models', {}) if config else {}
    
    def get_default_model(self, provider: str) -> Optional[str]:
        """Get default model for a provider."""
        config = self.get_config(provider)
        return config.get('default_model') if config else None
    
    def set_api_key(self, provider: str, api_key: str):
        """Set API key for a provider."""
        if provider in self.configs:
            self.configs[provider]['api_key'] = api_key
            self.configs[provider]['enabled'] = bool(api_key)
            logger.info(f"API key set for {provider}")
    
    def validate_config(self, provider: str) -> tuple[bool, str]:
        """Validate configuration for a provider."""
        config = self.get_config(provider)
        if not config:
            return False, f"Provider '{provider}' not found"
        
        if not config.get('enabled', False):
            return False, f"Provider '{provider}' is not enabled"
        
        # Check API key requirement
        if provider not in ['ollama', 'lmstudio'] and not config.get('api_key'):
            return False, f"API key required for {provider}"
        
        return True, "Configuration valid"
    
    def get_setup_instructions(self, provider: str) -> str:
        """Get setup instructions for a provider."""
        instructions = {
            'groq': """
1. Visit: https://console.groq.com
2. Create free account
3. Generate API key
4. Set environment variable: GROQ_API_KEY=your_key
5. Restart application
""",
            'openai': """
1. Visit: https://platform.openai.com
2. Create account and add payment method
3. Generate API key
4. Set environment variable: OPENAI_API_KEY=your_key
5. Restart application
""",
            'anthropic': """
1. Visit: https://console.anthropic.com
2. Create account
3. Generate API key
4. Set environment variable: ANTHROPIC_API_KEY=your_key
5. Restart application
""",
            'google': """
1. Visit: https://makersuite.google.com
2. Create account
3. Generate API key
4. Set environment variable: GOOGLE_API_KEY=your_key
5. Restart application
""",
            'ollama': """
1. Download: https://ollama.ai
2. Install Ollama
3. Pull model: ollama pull llama3
4. Start server: ollama serve
5. No API key needed
""",
            'huggingface': """
1. Visit: https://huggingface.co
2. Create account
3. Generate API token
4. Set environment variable: HUGGINGFACE_API_KEY=your_token
5. Restart application
"""
        }
        return instructions.get(provider, "No setup instructions available")


# Global API configuration instance
api_config = APIConfig()
