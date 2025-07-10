"""
Configuration management for CV evaluation system.
"""

import os
from pathlib import Path
from typing import Dict, Any, Optional
from dotenv import load_dotenv
import yaml


class Config:
    """Configuration manager for the CV evaluation system."""
    
    def __init__(self, config_file: Optional[str] = None):
        # Load environment variables
        load_dotenv()
        
        # Default configuration
        self._config = {
            'app': {
                'name': os.getenv('APP_NAME', 'CV Evaluation System'),
                'version': os.getenv('APP_VERSION', '1.0.0'),
                'debug': os.getenv('DEBUG', 'False').lower() == 'true'
            },
            'evaluation': {
                'default_language': os.getenv('DEFAULT_LANGUAGE', 'en'),
                'scoring_scale': int(os.getenv('SCORING_SCALE', '100')),
                'min_score_threshold': int(os.getenv('MIN_SCORE_THRESHOLD', '50'))
            },
            'openai': {
                'api_key': os.getenv('OPENAI_API_KEY'),
                'model': os.getenv('OPENAI_MODEL', 'gpt-3.5-turbo'),
                'max_tokens': int(os.getenv('OPENAI_MAX_TOKENS', '2000'))
            },
            'files': {
                'max_file_size_mb': int(os.getenv('MAX_FILE_SIZE_MB', '10')),
                'supported_formats': os.getenv('SUPPORTED_FORMATS', 'pdf').split(','),
                'temp_dir': os.getenv('TEMP_DIR', 'temp/')
            },
            'batch': {
                'max_concurrent_jobs': int(os.getenv('MAX_CONCURRENT_JOBS', '5')),
                'timeout': int(os.getenv('BATCH_TIMEOUT', '300'))
            },
            'reports': {
                'template': os.getenv('REPORT_TEMPLATE', 'default'),
                'output_format': os.getenv('OUTPUT_FORMAT', 'pdf'),
                'include_charts': os.getenv('INCLUDE_CHARTS', 'True').lower() == 'true'
            },
            'logging': {
                'level': os.getenv('LOG_LEVEL', 'INFO'),
                'file': os.getenv('LOG_FILE', 'logs/cv_evaluator.log')
            }
        }
        
        # Load additional configuration from file if provided
        if config_file:
            self.load_config_file(config_file)
    
    def load_config_file(self, config_file: str):
        """Load configuration from YAML file."""
        config_path = Path(config_file)
        if config_path.exists():
            with open(config_path, 'r', encoding='utf-8') as f:
                file_config = yaml.safe_load(f)
                self._merge_config(self._config, file_config)
    
    def _merge_config(self, base: Dict[str, Any], override: Dict[str, Any]):
        """Recursively merge configuration dictionaries."""
        for key, value in override.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                self._merge_config(base[key], value)
            else:
                base[key] = value
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value using dot notation."""
        keys = key.split('.')
        value = self._config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def set(self, key: str, value: Any):
        """Set configuration value using dot notation."""
        keys = key.split('.')
        config = self._config
        
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        config[keys[-1]] = value
    
    @property
    def app_name(self) -> str:
        return self.get('app.name')
    
    @property
    def app_version(self) -> str:
        return self.get('app.version')
    
    @property
    def debug(self) -> bool:
        return self.get('app.debug')
    
    @property
    def openai_api_key(self) -> Optional[str]:
        return self.get('openai.api_key')
    
    @property
    def scoring_scale(self) -> int:
        return self.get('evaluation.scoring_scale')
    
    @property
    def temp_dir(self) -> str:
        return self.get('files.temp_dir')
    
    @property
    def log_level(self) -> str:
        return self.get('logging.level')
    
    @property
    def log_file(self) -> str:
        return self.get('logging.file')


# Global configuration instance
config = Config()
