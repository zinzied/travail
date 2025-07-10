"""
CV scoring algorithms and evaluation logic.
"""

import logging
from typing import List, Dict, Any, NamedTuple
from datetime import datetime
from ..core.models import (
    Skill, WorkExperience, Education, EvaluationCriteria,
    SkillLevel, EducationLevel
)

logger = logging.getLogger(__name__)


class ScoreResult(NamedTuple):
    """Result of a scoring operation."""
    score: float
    max_score: float
    details: Dict[str, Any]
    feedback: str


class CVScorer:
    """Scoring engine for CV evaluation."""
    
    def __init__(self, evaluation_criteria: EvaluationCriteria):
        self.criteria = evaluation_criteria
        
        # Skill level weights
        self.skill_level_weights = {
            SkillLevel.BEGINNER: 0.25,
            SkillLevel.INTERMEDIATE: 0.5,
            SkillLevel.ADVANCED: 0.75,
            SkillLevel.EXPERT: 1.0
        }
        
        # Education level weights
        self.education_level_weights = {
            EducationLevel.HIGH_SCHOOL: 0.3,
            EducationLevel.CERTIFICATION: 0.4,
            EducationLevel.BACHELOR: 0.7,
            EducationLevel.MASTER: 0.9,
            EducationLevel.PHD: 1.0
        }
    
    def score_skills(self, skills: List[Skill]) -> ScoreResult:
        """
        Score the skills section of a CV.
        
        Args:
            skills: List of candidate skills
            
        Returns:
            ScoreResult with skills evaluation
        """
        max_score = 100.0
        
        if not skills:
            return ScoreResult(
                score=0.0,
                max_score=max_score,
                details={"skill_count": 0, "required_skills_found": 0},
                feedback="No skills identified in CV"
            )
        
        # Calculate required skills score (60% of total)
        required_score = self._score_required_skills(skills) * 0.6
        
        # Calculate preferred skills score (20% of total)
        preferred_score = self._score_preferred_skills(skills) * 0.2
        
        # Calculate skill diversity score (10% of total)
        diversity_score = self._score_skill_diversity(skills) * 0.1
        
        # Calculate skill level score (10% of total)
        level_score = self._score_skill_levels(skills) * 0.1
        
        total_score = (required_score + preferred_score + diversity_score + level_score) * max_score
        
        # Count required skills found
        required_skills_lower = [skill.lower() for skill in self.criteria.required_skills]
        candidate_skills_lower = [skill.name.lower() for skill in skills]
        required_found = len(set(required_skills_lower) & set(candidate_skills_lower))
        
        details = {
            "skill_count": len(skills),
            "required_skills_found": required_found,
            "required_skills_total": len(self.criteria.required_skills),
            "preferred_skills_found": self._count_preferred_skills(skills),
            "skill_categories": self._categorize_skills(skills)
        }
        
        feedback = self._generate_skills_feedback(skills, details)
        
        return ScoreResult(
            score=min(total_score, max_score),
            max_score=max_score,
            details=details,
            feedback=feedback
        )
    
    def score_experience(self, work_experience: List[WorkExperience]) -> ScoreResult:
        """Score the work experience section."""
        max_score = 100.0
        
        if not work_experience:
            return ScoreResult(
                score=0.0,
                max_score=max_score,
                details={"total_experience_years": 0, "job_count": 0},
                feedback="No work experience provided"
            )
        
        # Calculate total experience in years
        total_months = sum(exp.duration_months or 0 for exp in work_experience)
        total_years = total_months / 12
        
        # Experience duration score (50% of total)
        duration_score = self._score_experience_duration(total_years) * 0.5
        
        # Experience relevance score (30% of total)
        relevance_score = self._score_experience_relevance(work_experience) * 0.3
        
        # Career progression score (20% of total)
        progression_score = self._score_career_progression(work_experience) * 0.2
        
        total_score = (duration_score + relevance_score + progression_score) * max_score
        
        details = {
            "total_experience_years": round(total_years, 1),
            "job_count": len(work_experience),
            "min_required_years": self.criteria.min_experience_years,
            "companies": [exp.company for exp in work_experience],
            "positions": [exp.position for exp in work_experience]
        }
        
        feedback = self._generate_experience_feedback(work_experience, details)
        
        return ScoreResult(
            score=min(total_score, max_score),
            max_score=max_score,
            details=details,
            feedback=feedback
        )
    
    def score_education(self, education: List[Education]) -> ScoreResult:
        """Score the education section."""
        max_score = 100.0
        
        if not education:
            return ScoreResult(
                score=0.0,
                max_score=max_score,
                details={"education_count": 0, "highest_level": None},
                feedback="No education information provided"
            )
        
        # Find highest education level
        highest_level = self._get_highest_education_level(education)
        
        # Education level score (70% of total)
        level_score = self._score_education_level(highest_level) * 0.7
        
        # Education relevance score (20% of total)
        relevance_score = self._score_education_relevance(education) * 0.2
        
        # Education prestige/quality score (10% of total)
        quality_score = self._score_education_quality(education) * 0.1
        
        total_score = (level_score + relevance_score + quality_score) * max_score
        
        details = {
            "education_count": len(education),
            "highest_level": highest_level.value if highest_level else None,
            "institutions": [edu.institution for edu in education],
            "degrees": [edu.degree for edu in education],
            "fields": [edu.field_of_study for edu in education if edu.field_of_study]
        }
        
        feedback = self._generate_education_feedback(education, details)
        
        return ScoreResult(
            score=min(total_score, max_score),
            max_score=max_score,
            details=details,
            feedback=feedback
        )
    
    def score_additional_factors(self, languages: List[str], certifications: List[str], 
                               projects: List[str]) -> ScoreResult:
        """Score additional factors like languages, certifications, and projects."""
        max_score = 100.0
        
        # Languages score (30% of total)
        languages_score = min(len(languages) / 3, 1.0) * 0.3
        
        # Certifications score (40% of total)
        certifications_score = min(len(certifications) / 5, 1.0) * 0.4
        
        # Projects score (30% of total)
        projects_score = min(len(projects) / 3, 1.0) * 0.3
        
        total_score = (languages_score + certifications_score + projects_score) * max_score
        
        details = {
            "languages_count": len(languages),
            "certifications_count": len(certifications),
            "projects_count": len(projects),
            "languages": languages,
            "certifications": certifications[:5],  # Limit display
            "projects": projects[:5]  # Limit display
        }
        
        feedback = self._generate_additional_feedback(details)
        
        return ScoreResult(
            score=min(total_score, max_score),
            max_score=max_score,
            details=details,
            feedback=feedback
        )
    
    def _score_required_skills(self, skills: List[Skill]) -> float:
        """Score based on required skills match."""
        if not self.criteria.required_skills:
            return 1.0
        
        required_skills_lower = [skill.lower() for skill in self.criteria.required_skills]
        candidate_skills_lower = [skill.name.lower() for skill in skills]
        
        matches = len(set(required_skills_lower) & set(candidate_skills_lower))
        return matches / len(self.criteria.required_skills)
    
    def _score_preferred_skills(self, skills: List[Skill]) -> float:
        """Score based on preferred skills match."""
        if not self.criteria.preferred_skills:
            return 1.0
        
        preferred_skills_lower = [skill.lower() for skill in self.criteria.preferred_skills]
        candidate_skills_lower = [skill.name.lower() for skill in skills]
        
        matches = len(set(preferred_skills_lower) & set(candidate_skills_lower))
        return min(matches / len(self.criteria.preferred_skills), 1.0)
    
    def _score_skill_diversity(self, skills: List[Skill]) -> float:
        """Score based on skill diversity."""
        categories = set(skill.category for skill in skills if skill.category)
        return min(len(categories) / 5, 1.0)  # Normalize to max 5 categories
    
    def _score_skill_levels(self, skills: List[Skill]) -> float:
        """Score based on skill proficiency levels."""
        if not skills:
            return 0.0
        
        total_weight = 0.0
        skill_count = 0
        
        for skill in skills:
            if skill.level:
                weight = self.skill_level_weights.get(skill.level, 0.5)
                total_weight += weight
                skill_count += 1
        
        return (total_weight / skill_count) if skill_count > 0 else 0.5
    
    def _score_experience_duration(self, total_years: float) -> float:
        """Score based on total experience duration."""
        if self.criteria.min_experience_years == 0:
            return 1.0
        
        return min(total_years / self.criteria.min_experience_years, 1.0)
    
    def _score_experience_relevance(self, work_experience: List[WorkExperience]) -> float:
        """Score based on experience relevance to industry keywords."""
        if not self.criteria.industry_keywords:
            return 1.0
        
        relevance_score = 0.0
        for exp in work_experience:
            exp_text = f"{exp.position} {exp.description or ''}".lower()
            keyword_matches = sum(1 for keyword in self.criteria.industry_keywords 
                                if keyword.lower() in exp_text)
            relevance_score += keyword_matches / len(self.criteria.industry_keywords)
        
        return min(relevance_score / len(work_experience), 1.0) if work_experience else 0.0
    
    def _score_career_progression(self, work_experience: List[WorkExperience]) -> float:
        """Score based on career progression indicators."""
        if len(work_experience) < 2:
            return 0.5
        
        # Simple progression scoring based on position titles
        progression_indicators = ['senior', 'lead', 'manager', 'director', 'head', 'chief']
        progression_score = 0.0
        
        for exp in work_experience:
            position_lower = exp.position.lower()
            for i, indicator in enumerate(progression_indicators):
                if indicator in position_lower:
                    progression_score += (i + 1) / len(progression_indicators)
                    break
        
        return min(progression_score / len(work_experience), 1.0)
    
    def _get_highest_education_level(self, education: List[Education]) -> EducationLevel:
        """Get the highest education level from the list."""
        level_order = [EducationLevel.HIGH_SCHOOL, EducationLevel.CERTIFICATION, 
                      EducationLevel.BACHELOR, EducationLevel.MASTER, EducationLevel.PHD]
        
        highest = EducationLevel.HIGH_SCHOOL
        for edu in education:
            if edu.level and level_order.index(edu.level) > level_order.index(highest):
                highest = edu.level
        
        return highest
    
    def _score_education_level(self, highest_level: EducationLevel) -> float:
        """Score based on education level."""
        return self.education_level_weights.get(highest_level, 0.3)
    
    def _score_education_relevance(self, education: List[Education]) -> float:
        """Score based on education field relevance."""
        if not self.criteria.industry_keywords:
            return 1.0
        
        relevance_score = 0.0
        for edu in education:
            field_text = f"{edu.field_of_study or ''} {edu.degree}".lower()
            keyword_matches = sum(1 for keyword in self.criteria.industry_keywords 
                                if keyword.lower() in field_text)
            relevance_score += keyword_matches / len(self.criteria.industry_keywords)
        
        return min(relevance_score / len(education), 1.0) if education else 0.0
    
    def _score_education_quality(self, education: List[Education]) -> float:
        """Score based on education quality indicators."""
        # Simple quality scoring based on institution keywords
        quality_indicators = ['university', 'institute', 'college']
        quality_score = 0.0
        
        for edu in education:
            institution_lower = edu.institution.lower()
            if any(indicator in institution_lower for indicator in quality_indicators):
                quality_score += 1.0
        
        return min(quality_score / len(education), 1.0) if education else 0.0
    
    def _count_preferred_skills(self, skills: List[Skill]) -> int:
        """Count how many preferred skills are found."""
        if not self.criteria.preferred_skills:
            return 0
        
        preferred_skills_lower = [skill.lower() for skill in self.criteria.preferred_skills]
        candidate_skills_lower = [skill.name.lower() for skill in skills]
        
        return len(set(preferred_skills_lower) & set(candidate_skills_lower))
    
    def _categorize_skills(self, skills: List[Skill]) -> Dict[str, int]:
        """Categorize skills and count by category."""
        categories = {}
        for skill in skills:
            category = skill.category or "general"
            categories[category] = categories.get(category, 0) + 1
        return categories
    
    def _generate_skills_feedback(self, skills: List[Skill], details: Dict[str, Any]) -> str:
        """Generate feedback for skills section."""
        feedback_parts = []
        
        required_found = details["required_skills_found"]
        required_total = details["required_skills_total"]
        
        if required_found == required_total:
            feedback_parts.append("All required skills are present")
        elif required_found > 0:
            feedback_parts.append(f"{required_found}/{required_total} required skills found")
        else:
            feedback_parts.append("No required skills identified")
        
        if details["preferred_skills_found"] > 0:
            feedback_parts.append(f"{details['preferred_skills_found']} preferred skills found")
        
        skill_count = details["skill_count"]
        if skill_count >= 10:
            feedback_parts.append("Good skill diversity")
        elif skill_count >= 5:
            feedback_parts.append("Moderate skill set")
        else:
            feedback_parts.append("Limited skill set")
        
        return ". ".join(feedback_parts)
    
    def _generate_experience_feedback(self, work_experience: List[WorkExperience], 
                                    details: Dict[str, Any]) -> str:
        """Generate feedback for experience section."""
        feedback_parts = []
        
        total_years = details["total_experience_years"]
        min_required = details["min_required_years"]
        
        if total_years >= min_required:
            feedback_parts.append(f"Meets experience requirement ({total_years} years)")
        else:
            feedback_parts.append(f"Below minimum experience ({total_years}/{min_required} years)")
        
        job_count = details["job_count"]
        if job_count >= 3:
            feedback_parts.append("Good experience diversity")
        elif job_count == 2:
            feedback_parts.append("Moderate experience diversity")
        else:
            feedback_parts.append("Limited experience diversity")
        
        return ". ".join(feedback_parts)
    
    def _generate_education_feedback(self, education: List[Education], 
                                   details: Dict[str, Any]) -> str:
        """Generate feedback for education section."""
        feedback_parts = []
        
        highest_level = details["highest_level"]
        if highest_level:
            feedback_parts.append(f"Highest education: {highest_level.replace('_', ' ').title()}")
        
        education_count = details["education_count"]
        if education_count >= 2:
            feedback_parts.append("Multiple educational qualifications")
        
        return ". ".join(feedback_parts) if feedback_parts else "Basic education information"
    
    def _generate_additional_feedback(self, details: Dict[str, Any]) -> str:
        """Generate feedback for additional factors."""
        feedback_parts = []
        
        if details["languages_count"] > 1:
            feedback_parts.append(f"Multilingual ({details['languages_count']} languages)")
        
        if details["certifications_count"] >= 3:
            feedback_parts.append("Well-certified professional")
        elif details["certifications_count"] > 0:
            feedback_parts.append("Some relevant certifications")
        
        if details["projects_count"] >= 3:
            feedback_parts.append("Strong project portfolio")
        elif details["projects_count"] > 0:
            feedback_parts.append("Some project experience")
        
        return ". ".join(feedback_parts) if feedback_parts else "Limited additional information"
