"""
Evaluation criteria loader and manager.
"""

import yaml
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List
from .models import EvaluationCriteria, EducationLevel
from ..utils.exceptions import ConfigurationError

logger = logging.getLogger(__name__)


class CriteriaLoader:
    """Load and manage evaluation criteria from configuration files."""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path or "config/evaluation_criteria.yaml"
        self._criteria_cache = {}
    
    def load_criteria(self, job_template: Optional[str] = None) -> EvaluationCriteria:
        """
        Load evaluation criteria from configuration.
        
        Args:
            job_template: Optional job template name (e.g., 'software_engineer')
            
        Returns:
            EvaluationCriteria object
        """
        cache_key = job_template or "default"
        
        if cache_key in self._criteria_cache:
            return self._criteria_cache[cache_key]
        
        try:
            config = self._load_config_file()
            criteria = self._build_criteria(config, job_template)
            self._criteria_cache[cache_key] = criteria
            
            logger.info(f"Loaded evaluation criteria for: {cache_key}")
            return criteria
            
        except Exception as e:
            logger.error(f"Failed to load evaluation criteria: {e}")
            raise ConfigurationError(f"Failed to load evaluation criteria: {e}")
    
    def get_available_templates(self) -> List[str]:
        """Get list of available job templates."""
        try:
            config = self._load_config_file()
            return list(config.get('job_templates', {}).keys())
        except Exception:
            return []
    
    def _load_config_file(self) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        config_path = Path(self.config_path)
        
        if not config_path.exists():
            raise ConfigurationError(f"Configuration file not found: {config_path}")
        
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            
            if not isinstance(config, dict):
                raise ConfigurationError("Invalid configuration format")
            
            return config
            
        except yaml.YAMLError as e:
            raise ConfigurationError(f"Invalid YAML format: {e}")
        except Exception as e:
            raise ConfigurationError(f"Failed to read configuration file: {e}")
    
    def _build_criteria(self, config: Dict[str, Any], job_template: Optional[str]) -> EvaluationCriteria:
        """Build EvaluationCriteria object from configuration."""
        # Start with default configuration
        criteria_data = {
            'max_score': config.get('max_score', 100),
            'scoring_weights': config.get('scoring_weights', {
                'skills': 0.4,
                'experience': 0.3,
                'education': 0.2,
                'additional': 0.1
            })
        }
        
        # Load skills configuration
        skills_config = config.get('skills', {})
        criteria_data['required_skills'] = skills_config.get('required', [])
        criteria_data['preferred_skills'] = skills_config.get('preferred', [])
        
        # Load experience configuration
        experience_config = config.get('experience', {})
        criteria_data['min_experience_years'] = experience_config.get('min_years', 0)
        criteria_data['industry_keywords'] = experience_config.get('industry_keywords', [])
        
        # Load education configuration
        education_config = config.get('education', {})
        required_level = education_config.get('required_level')
        if required_level:
            try:
                criteria_data['required_education_level'] = EducationLevel(required_level)
            except ValueError:
                logger.warning(f"Invalid education level: {required_level}")
        
        # Override with job template if specified
        if job_template:
            template_config = config.get('job_templates', {}).get(job_template, {})
            if template_config:
                criteria_data.update(self._merge_template_config(criteria_data, template_config))
                logger.info(f"Applied job template: {job_template}")
        
        return EvaluationCriteria(**criteria_data)
    
    def _merge_template_config(self, base_config: Dict[str, Any], 
                             template_config: Dict[str, Any]) -> Dict[str, Any]:
        """Merge job template configuration with base configuration."""
        merged = base_config.copy()
        
        # Override specific fields from template
        template_fields = [
            'required_skills', 'preferred_skills', 'min_experience_years',
            'industry_keywords', 'required_education_level'
        ]
        
        for field in template_fields:
            if field in template_config:
                merged[field] = template_config[field]
        
        return merged
    
    def validate_criteria(self, criteria: EvaluationCriteria) -> bool:
        """
        Validate evaluation criteria.
        
        Args:
            criteria: EvaluationCriteria to validate
            
        Returns:
            True if valid, raises ConfigurationError if invalid
        """
        # Check scoring weights sum to 1.0
        total_weight = sum(criteria.scoring_weights.values())
        if abs(total_weight - 1.0) > 0.01:
            raise ConfigurationError(f"Scoring weights must sum to 1.0, got {total_weight}")
        
        # Check max score is positive
        if criteria.max_score <= 0:
            raise ConfigurationError("Max score must be positive")
        
        # Check min experience years is non-negative
        if criteria.min_experience_years < 0:
            raise ConfigurationError("Min experience years must be non-negative")
        
        logger.info("Evaluation criteria validation passed")
        return True


class CriteriaManager:
    """Manage multiple evaluation criteria configurations."""
    
    def __init__(self, config_dir: str = "config"):
        self.config_dir = Path(config_dir)
        self.loaders = {}
    
    def get_criteria(self, criteria_name: str = "default", 
                    job_template: Optional[str] = None) -> EvaluationCriteria:
        """
        Get evaluation criteria by name and optional job template.
        
        Args:
            criteria_name: Name of criteria configuration file
            job_template: Optional job template name
            
        Returns:
            EvaluationCriteria object
        """
        if criteria_name not in self.loaders:
            config_file = self.config_dir / f"{criteria_name}.yaml"
            if not config_file.exists():
                config_file = self.config_dir / "evaluation_criteria.yaml"
            
            self.loaders[criteria_name] = CriteriaLoader(str(config_file))
        
        return self.loaders[criteria_name].load_criteria(job_template)
    
    def list_available_criteria(self) -> List[str]:
        """List available criteria configuration files."""
        criteria_files = []
        
        for yaml_file in self.config_dir.glob("*.yaml"):
            if yaml_file.stem not in ['config', 'settings']:
                criteria_files.append(yaml_file.stem)
        
        return criteria_files
    
    def list_job_templates(self, criteria_name: str = "default") -> List[str]:
        """List available job templates for a criteria configuration."""
        try:
            loader = CriteriaLoader(str(self.config_dir / f"{criteria_name}.yaml"))
            return loader.get_available_templates()
        except Exception:
            return []


# Global criteria manager instance
criteria_manager = CriteriaManager()
