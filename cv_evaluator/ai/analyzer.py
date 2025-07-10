"""
Main CV analysis engine using AI and NLP techniques.
"""

import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
from ..core.models import (
    CVData, CVAnalysisResult, EvaluationCriteria, SectionScore,
    Skill, WorkExperience, Education
)
from .scorer import CVScorer
from .nlp_processor import NLPProcessor
from ..utils.exceptions import AnalysisError

logger = logging.getLogger(__name__)


class CVAnalyzer:
    """Main CV analysis engine that coordinates all analysis components."""
    
    def __init__(self, evaluation_criteria: Optional[EvaluationCriteria] = None):
        self.evaluation_criteria = evaluation_criteria or self._get_default_criteria()
        self.scorer = CVScorer(self.evaluation_criteria)
        self.nlp_processor = NLPProcessor()
        
    def analyze_cv(self, cv_data: CVData) -> CVAnalysisResult:
        """
        Perform comprehensive CV analysis.
        
        Args:
            cv_data: Structured CV data
            
        Returns:
            Complete analysis result with scores and recommendations
        """
        logger.info("Starting comprehensive CV analysis")
        
        try:
            # Enhance CV data with NLP analysis
            enhanced_cv_data = self._enhance_cv_data(cv_data)
            
            # Calculate section scores
            section_scores = self._calculate_section_scores(enhanced_cv_data)
            
            # Calculate overall score
            overall_score = self._calculate_overall_score(section_scores)
            
            # Generate insights
            strengths = self._identify_strengths(enhanced_cv_data, section_scores)
            weaknesses = self._identify_weaknesses(enhanced_cv_data, section_scores)
            recommendations = self._generate_recommendations(enhanced_cv_data, weaknesses)
            
            # Calculate fit percentage
            fit_percentage = self._calculate_fit_percentage(enhanced_cv_data)
            
            result = CVAnalysisResult(
                cv_data=enhanced_cv_data,
                overall_score=overall_score,
                section_scores=section_scores,
                strengths=strengths,
                weaknesses=weaknesses,
                recommendations=recommendations,
                fit_percentage=fit_percentage,
                evaluation_criteria=self.evaluation_criteria,
                analysis_timestamp=datetime.now()
            )
            
            logger.info(f"CV analysis completed. Overall score: {overall_score:.1f}")
            return result
            
        except Exception as e:
            logger.error(f"CV analysis failed: {e}")
            raise AnalysisError(f"Failed to analyze CV: {e}")
    
    def _enhance_cv_data(self, cv_data: CVData) -> CVData:
        """Enhance CV data with additional NLP analysis."""
        # Enhance skills with better categorization and level detection
        enhanced_skills = self.nlp_processor.enhance_skills(cv_data.skills, cv_data.raw_text)
        
        # Extract additional skills from work experience descriptions
        experience_skills = self.nlp_processor.extract_skills_from_experience(cv_data.work_experience)
        
        # Merge and deduplicate skills
        all_skills = self._merge_skills(enhanced_skills, experience_skills)
        
        # Enhance work experience with better parsing
        enhanced_experience = self.nlp_processor.enhance_work_experience(cv_data.work_experience)
        
        # Create enhanced CV data
        enhanced_cv_data = cv_data.model_copy()
        enhanced_cv_data.skills = all_skills
        enhanced_cv_data.work_experience = enhanced_experience
        
        return enhanced_cv_data
    
    def _calculate_section_scores(self, cv_data: CVData) -> List[SectionScore]:
        """Calculate scores for each CV section."""
        section_scores = []
        
        # Skills score
        skills_score = self.scorer.score_skills(cv_data.skills)
        section_scores.append(SectionScore(
            section="skills",
            score=skills_score.score,
            max_score=skills_score.max_score,
            details=skills_score.details,
            feedback=skills_score.feedback
        ))
        
        # Experience score
        experience_score = self.scorer.score_experience(cv_data.work_experience)
        section_scores.append(SectionScore(
            section="experience",
            score=experience_score.score,
            max_score=experience_score.max_score,
            details=experience_score.details,
            feedback=experience_score.feedback
        ))
        
        # Education score
        education_score = self.scorer.score_education(cv_data.education)
        section_scores.append(SectionScore(
            section="education",
            score=education_score.score,
            max_score=education_score.max_score,
            details=education_score.details,
            feedback=education_score.feedback
        ))
        
        # Additional factors score (languages, certifications, projects)
        additional_score = self.scorer.score_additional_factors(
            cv_data.languages, cv_data.certifications, cv_data.projects
        )
        section_scores.append(SectionScore(
            section="additional",
            score=additional_score.score,
            max_score=additional_score.max_score,
            details=additional_score.details,
            feedback=additional_score.feedback
        ))
        
        return section_scores
    
    def _calculate_overall_score(self, section_scores: List[SectionScore]) -> float:
        """Calculate weighted overall score."""
        total_weighted_score = 0.0
        total_weight = 0.0
        
        for section_score in section_scores:
            weight = self.evaluation_criteria.scoring_weights.get(section_score.section, 0.0)
            weighted_score = (section_score.score / section_score.max_score) * weight * self.evaluation_criteria.max_score
            total_weighted_score += weighted_score
            total_weight += weight
        
        if total_weight == 0:
            return 0.0
        
        return min(total_weighted_score, self.evaluation_criteria.max_score)
    
    def _identify_strengths(self, cv_data: CVData, section_scores: List[SectionScore]) -> List[str]:
        """Identify candidate's strengths based on analysis."""
        strengths = []
        
        # High-scoring sections
        for section_score in section_scores:
            score_percentage = (section_score.score / section_score.max_score) * 100
            if score_percentage >= 80:
                strengths.append(f"Strong {section_score.section} profile with {score_percentage:.0f}% score")
        
        # Specific strengths
        if len(cv_data.skills) >= 10:
            strengths.append(f"Diverse skill set with {len(cv_data.skills)} identified skills")
        
        if cv_data.work_experience:
            total_experience = sum(exp.duration_months or 0 for exp in cv_data.work_experience) / 12
            if total_experience >= 5:
                strengths.append(f"Extensive work experience ({total_experience:.1f} years)")
        
        if len(cv_data.education) >= 2:
            strengths.append("Strong educational background with multiple qualifications")
        
        if len(cv_data.certifications) >= 3:
            strengths.append(f"Well-certified professional with {len(cv_data.certifications)} certifications")
        
        return strengths[:5]  # Limit to top 5 strengths
    
    def _identify_weaknesses(self, cv_data: CVData, section_scores: List[SectionScore]) -> List[str]:
        """Identify areas for improvement."""
        weaknesses = []
        
        # Low-scoring sections
        for section_score in section_scores:
            score_percentage = (section_score.score / section_score.max_score) * 100
            if score_percentage < 50:
                weaknesses.append(f"Limited {section_score.section} profile ({score_percentage:.0f}% score)")
        
        # Missing required skills
        required_skills = set(skill.lower() for skill in self.evaluation_criteria.required_skills)
        candidate_skills = set(skill.name.lower() for skill in cv_data.skills)
        missing_skills = required_skills - candidate_skills
        
        if missing_skills:
            weaknesses.append(f"Missing required skills: {', '.join(list(missing_skills)[:3])}")
        
        # Experience gaps
        if not cv_data.work_experience:
            weaknesses.append("No work experience provided")
        elif len(cv_data.work_experience) == 1:
            weaknesses.append("Limited work experience diversity")
        
        # Education gaps
        if not cv_data.education:
            weaknesses.append("No educational background provided")
        
        return weaknesses[:5]  # Limit to top 5 weaknesses
    
    def _generate_recommendations(self, cv_data: CVData, weaknesses: List[str]) -> List[str]:
        """Generate actionable recommendations."""
        recommendations = []
        
        # Skill-based recommendations
        required_skills = set(skill.lower() for skill in self.evaluation_criteria.required_skills)
        candidate_skills = set(skill.name.lower() for skill in cv_data.skills)
        missing_skills = required_skills - candidate_skills
        
        if missing_skills:
            recommendations.append(f"Consider acquiring skills in: {', '.join(list(missing_skills)[:3])}")
        
        # Experience recommendations
        if not cv_data.work_experience:
            recommendations.append("Gain relevant work experience through internships or entry-level positions")
        
        # Education recommendations
        if not cv_data.education and self.evaluation_criteria.required_education_level:
            recommendations.append(f"Consider pursuing {self.evaluation_criteria.required_education_level.value} education")
        
        # Certification recommendations
        if len(cv_data.certifications) < 2:
            recommendations.append("Obtain industry-relevant certifications to strengthen profile")
        
        # General recommendations
        if cv_data.extraction_confidence < 0.7:
            recommendations.append("Improve CV formatting and structure for better readability")
        
        return recommendations[:5]  # Limit to top 5 recommendations
    
    def _calculate_fit_percentage(self, cv_data: CVData) -> float:
        """Calculate how well the candidate fits the criteria."""
        fit_factors = []
        
        # Skills fit
        required_skills = set(skill.lower() for skill in self.evaluation_criteria.required_skills)
        candidate_skills = set(skill.name.lower() for skill in cv_data.skills)
        
        if required_skills:
            skills_fit = len(required_skills & candidate_skills) / len(required_skills)
            fit_factors.append(skills_fit)
        
        # Experience fit
        if self.evaluation_criteria.min_experience_years > 0:
            total_experience = sum(exp.duration_months or 0 for exp in cv_data.work_experience) / 12
            experience_fit = min(total_experience / self.evaluation_criteria.min_experience_years, 1.0)
            fit_factors.append(experience_fit)
        
        # Education fit
        if self.evaluation_criteria.required_education_level and cv_data.education:
            # Simplified education level comparison
            education_fit = 1.0 if cv_data.education else 0.5
            fit_factors.append(education_fit)
        
        return (sum(fit_factors) / len(fit_factors) * 100) if fit_factors else 0.0
    
    def _merge_skills(self, skills1: List[Skill], skills2: List[Skill]) -> List[Skill]:
        """Merge and deduplicate skill lists."""
        skill_dict = {}
        
        # Add skills from first list
        for skill in skills1:
            skill_dict[skill.name.lower()] = skill
        
        # Add skills from second list, keeping higher confidence
        for skill in skills2:
            key = skill.name.lower()
            if key not in skill_dict or skill.confidence > skill_dict[key].confidence:
                skill_dict[key] = skill
        
        return list(skill_dict.values())
    
    def _get_default_criteria(self) -> EvaluationCriteria:
        """Get default evaluation criteria."""
        return EvaluationCriteria(
            required_skills=["python", "sql", "communication"],
            preferred_skills=["machine learning", "data analysis", "project management"],
            min_experience_years=2,
            scoring_weights={
                "skills": 0.4,
                "experience": 0.3,
                "education": 0.2,
                "additional": 0.1
            }
        )
