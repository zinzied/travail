"""
NLP processing utilities for CV analysis.
"""

import re
import logging
from typing import List, Dict, Set, Optional
from collections import Counter
from ..core.models import Skill, WorkExperience, SkillLevel

logger = logging.getLogger(__name__)


class NLPProcessor:
    """Natural Language Processing utilities for CV enhancement."""
    
    def __init__(self):
        # Skill level indicators
        self.skill_level_patterns = {
            SkillLevel.EXPERT: [
                r'expert', r'mastery', r'advanced', r'senior', r'lead',
                r'\d+\+?\s*years?', r'extensive', r'deep'
            ],
            SkillLevel.ADVANCED: [
                r'advanced', r'proficient', r'experienced', r'skilled',
                r'strong', r'solid', r'good'
            ],
            SkillLevel.INTERMEDIATE: [
                r'intermediate', r'moderate', r'working', r'familiar',
                r'some', r'basic'
            ],
            SkillLevel.BEGINNER: [
                r'beginner', r'novice', r'learning', r'basic',
                r'introductory', r'entry'
            ]
        }
        
        # Skill categories and keywords
        self.skill_categories = {
            'programming': {
                'python', 'java', 'javascript', 'c++', 'c#', 'php', 'ruby', 'go', 'rust',
                'scala', 'kotlin', 'swift', 'typescript', 'r', 'matlab', 'perl'
            },
            'web_development': {
                'html', 'css', 'react', 'angular', 'vue', 'node.js', 'express',
                'django', 'flask', 'spring', 'laravel', 'rails', 'asp.net'
            },
            'database': {
                'sql', 'mysql', 'postgresql', 'mongodb', 'redis', 'oracle',
                'sqlite', 'cassandra', 'elasticsearch', 'neo4j'
            },
            'cloud_devops': {
                'aws', 'azure', 'gcp', 'docker', 'kubernetes', 'jenkins',
                'terraform', 'ansible', 'chef', 'puppet', 'gitlab', 'circleci'
            },
            'data_science': {
                'machine learning', 'deep learning', 'data analysis', 'statistics',
                'pandas', 'numpy', 'scikit-learn', 'tensorflow', 'pytorch',
                'tableau', 'power bi', 'spark', 'hadoop'
            },
            'mobile': {
                'android', 'ios', 'react native', 'flutter', 'xamarin',
                'swift', 'kotlin', 'objective-c'
            },
            'tools': {
                'git', 'svn', 'jira', 'confluence', 'slack', 'trello',
                'visual studio', 'intellij', 'eclipse', 'vim', 'emacs'
            },
            'soft_skills': {
                'communication', 'leadership', 'teamwork', 'problem solving',
                'project management', 'agile', 'scrum', 'kanban'
            }
        }
        
        # Flatten skill keywords for easy lookup
        self.all_skills = set()
        for category_skills in self.skill_categories.values():
            self.all_skills.update(skill.lower() for skill in category_skills)
        
        # Experience level indicators
        self.experience_patterns = {
            'senior': r'senior|sr\.|lead|principal|chief|head|director',
            'mid': r'mid|middle|intermediate|experienced',
            'junior': r'junior|jr\.|entry|associate|trainee|intern'
        }
    
    def enhance_skills(self, skills: List[Skill], full_text: str) -> List[Skill]:
        """
        Enhance skills with better categorization and level detection.
        
        Args:
            skills: Initial list of skills
            full_text: Full CV text for context
            
        Returns:
            Enhanced list of skills
        """
        enhanced_skills = []
        
        for skill in skills:
            enhanced_skill = skill.model_copy()
            
            # Determine skill category if not set
            if not enhanced_skill.category:
                enhanced_skill.category = self._categorize_skill(skill.name)
            
            # Determine skill level if not set
            if not enhanced_skill.level:
                enhanced_skill.level = self._detect_skill_level(skill.name, full_text)
            
            # Extract years of experience if mentioned
            if not enhanced_skill.years_experience:
                enhanced_skill.years_experience = self._extract_years_experience(skill.name, full_text)
            
            # Improve confidence based on context
            enhanced_skill.confidence = self._calculate_skill_confidence(skill.name, full_text)
            
            enhanced_skills.append(enhanced_skill)
        
        return enhanced_skills
    
    def extract_skills_from_experience(self, work_experience: List[WorkExperience]) -> List[Skill]:
        """
        Extract additional skills from work experience descriptions.
        
        Args:
            work_experience: List of work experiences
            
        Returns:
            List of skills found in experience descriptions
        """
        extracted_skills = []
        
        for exp in work_experience:
            # Combine all text from experience
            exp_text = f"{exp.position} {exp.description or ''} {' '.join(exp.responsibilities or [])} {' '.join(exp.technologies or [])}"
            
            # Find skills in the text
            found_skills = self._find_skills_in_text(exp_text.lower())
            
            for skill_name in found_skills:
                # Create skill with context from experience
                skill = Skill(
                    name=skill_name,
                    category=self._categorize_skill(skill_name),
                    level=self._detect_skill_level(skill_name, exp_text),
                    confidence=0.7  # Medium confidence for extracted skills
                )
                extracted_skills.append(skill)
        
        return self._deduplicate_skills(extracted_skills)
    
    def enhance_work_experience(self, work_experience: List[WorkExperience]) -> List[WorkExperience]:
        """
        Enhance work experience with better parsing and analysis.
        
        Args:
            work_experience: List of work experiences
            
        Returns:
            Enhanced list of work experiences
        """
        enhanced_experience = []
        
        for exp in work_experience:
            enhanced_exp = exp.model_copy()
            
            # Extract duration if not set
            if not enhanced_exp.duration_months:
                enhanced_exp.duration_months = self._calculate_duration(exp.start_date, exp.end_date)
            
            # Extract technologies from description
            if not enhanced_exp.technologies:
                enhanced_exp.technologies = self._extract_technologies(exp.description or "")
            
            # Extract achievements from description
            if not enhanced_exp.achievements:
                enhanced_exp.achievements = self._extract_achievements(exp.description or "")
            
            enhanced_experience.append(enhanced_exp)
        
        return enhanced_experience
    
    def _categorize_skill(self, skill_name: str) -> str:
        """Categorize a skill based on predefined categories."""
        skill_lower = skill_name.lower()
        
        for category, skills in self.skill_categories.items():
            if skill_lower in skills:
                return category
        
        # Try partial matching for compound skills
        for category, skills in self.skill_categories.items():
            for known_skill in skills:
                if known_skill in skill_lower or skill_lower in known_skill:
                    return category
        
        return "general"
    
    def _detect_skill_level(self, skill_name: str, context: str) -> Optional[SkillLevel]:
        """Detect skill level from context."""
        context_lower = context.lower()
        skill_lower = skill_name.lower()
        
        # Look for skill level indicators near the skill mention
        skill_positions = [m.start() for m in re.finditer(re.escape(skill_lower), context_lower)]
        
        for pos in skill_positions:
            # Check surrounding text (50 characters before and after)
            start = max(0, pos - 50)
            end = min(len(context_lower), pos + len(skill_lower) + 50)
            surrounding_text = context_lower[start:end]
            
            # Check for level indicators
            for level, patterns in self.skill_level_patterns.items():
                for pattern in patterns:
                    if re.search(pattern, surrounding_text):
                        return level
        
        return None
    
    def _extract_years_experience(self, skill_name: str, context: str) -> Optional[int]:
        """Extract years of experience for a skill from context."""
        context_lower = context.lower()
        skill_lower = skill_name.lower()
        
        # Look for patterns like "5 years Python", "Python (3 years)", etc.
        patterns = [
            rf'{re.escape(skill_lower)}\s*\(?(\d+)\+?\s*years?\)?',
            rf'(\d+)\+?\s*years?\s*{re.escape(skill_lower)}',
            rf'{re.escape(skill_lower)}\s*[-–]\s*(\d+)\+?\s*years?'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, context_lower)
            if match:
                return int(match.group(1))
        
        return None
    
    def _calculate_skill_confidence(self, skill_name: str, context: str) -> float:
        """Calculate confidence score for skill extraction."""
        context_lower = context.lower()
        skill_lower = skill_name.lower()
        
        confidence = 0.5  # Base confidence
        
        # Increase confidence if skill is mentioned multiple times
        mentions = len(re.findall(re.escape(skill_lower), context_lower))
        confidence += min(mentions * 0.1, 0.3)
        
        # Increase confidence if skill is in known categories
        if skill_lower in self.all_skills:
            confidence += 0.2
        
        # Increase confidence if mentioned with level indicators
        for patterns in self.skill_level_patterns.values():
            for pattern in patterns:
                if re.search(f'{re.escape(skill_lower)}.*{pattern}|{pattern}.*{re.escape(skill_lower)}', context_lower):
                    confidence += 0.1
                    break
        
        return min(confidence, 1.0)
    
    def _find_skills_in_text(self, text: str) -> Set[str]:
        """Find known skills in text."""
        found_skills = set()
        
        # Direct matching
        for skill in self.all_skills:
            if skill in text:
                found_skills.add(skill)
        
        # Pattern-based extraction for common skill formats
        skill_patterns = [
            r'\b([A-Z][a-z]+(?:\.[a-z]+)*)\b',  # CamelCase or dotted notation
            r'\b([a-z]+(?:-[a-z]+)+)\b',        # Hyphenated skills
            r'\b([A-Z]{2,})\b'                  # Acronyms
        ]
        
        for pattern in skill_patterns:
            matches = re.findall(pattern, text)
            for match in matches:
                if len(match) > 2 and match.lower() not in {'and', 'the', 'for', 'with'}:
                    found_skills.add(match.lower())
        
        return found_skills
    
    def _deduplicate_skills(self, skills: List[Skill]) -> List[Skill]:
        """Remove duplicate skills, keeping the one with highest confidence."""
        skill_dict = {}
        
        for skill in skills:
            key = skill.name.lower()
            if key not in skill_dict or skill.confidence > skill_dict[key].confidence:
                skill_dict[key] = skill
        
        return list(skill_dict.values())
    
    def _calculate_duration(self, start_date: Optional[str], end_date: Optional[str]) -> Optional[int]:
        """Calculate duration in months from date strings."""
        if not start_date:
            return None
        
        # Simple year-based calculation
        try:
            start_year = int(re.search(r'\d{4}', start_date).group())
            
            if end_date and 'present' not in end_date.lower():
                end_year = int(re.search(r'\d{4}', end_date).group())
            else:
                end_year = 2024  # Current year
            
            return max((end_year - start_year) * 12, 1)
        except (AttributeError, ValueError):
            return None
    
    def _extract_technologies(self, description: str) -> List[str]:
        """Extract technologies from job description."""
        if not description:
            return []
        
        technologies = []
        description_lower = description.lower()
        
        # Find known technologies in description
        for skill in self.all_skills:
            if skill in description_lower:
                technologies.append(skill)
        
        return list(set(technologies))[:10]  # Limit to 10 technologies
    
    def _extract_achievements(self, description: str) -> List[str]:
        """Extract achievements from job description."""
        if not description:
            return []
        
        achievements = []
        
        # Look for achievement indicators
        achievement_patterns = [
            r'achieved\s+([^.]+)',
            r'improved\s+([^.]+)',
            r'increased\s+([^.]+)',
            r'reduced\s+([^.]+)',
            r'delivered\s+([^.]+)',
            r'led\s+([^.]+)',
            r'managed\s+([^.]+)'
        ]
        
        for pattern in achievement_patterns:
            matches = re.findall(pattern, description, re.IGNORECASE)
            achievements.extend(matches)
        
        # Clean and limit achievements
        cleaned_achievements = []
        for achievement in achievements:
            achievement = achievement.strip()
            if len(achievement) > 10 and len(achievement) < 200:
                cleaned_achievements.append(achievement)
        
        return cleaned_achievements[:5]  # Limit to 5 achievements
