"""
Main CV analysis engine using AI and NLP techniques.

Created by: Zied Boughdir (@zinzied)
GitHub: https://github.com/zinzied/cv-evaluation-system
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
from .free_models import get_ai_response, list_available_models, auto_select_ai_model
from ..utils.exceptions import AnalysisError

logger = logging.getLogger(__name__)


class CVAnalyzer:
    """Main CV analysis engine that coordinates all analysis components."""
    
    def __init__(self, evaluation_criteria: Optional[EvaluationCriteria] = None, use_ai: bool = True):
        self.evaluation_criteria = evaluation_criteria or self._get_default_criteria()
        self.scorer = CVScorer(self.evaluation_criteria)
        self.nlp_processor = NLPProcessor()
        self.use_ai = use_ai
        self.ai_model_available = False

        # Try to initialize AI model if requested
        if self.use_ai:
            selected_model = auto_select_ai_model()
            if selected_model:
                self.ai_model_available = True
                logger.info(f"CV Analyzer initialized with AI model: {selected_model}")
            else:
                logger.info("CV Analyzer initialized without AI (no models available)")
        else:
            logger.info("CV Analyzer initialized without AI (disabled)")
        
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

            # Enhance with AI if available
            if self.ai_model_available:
                try:
                    ai_insights = self._get_ai_insights(enhanced_cv_data, section_scores)
                    if ai_insights:
                        strengths.extend(ai_insights.get('strengths', []))
                        weaknesses.extend(ai_insights.get('weaknesses', []))
                        recommendations.extend(ai_insights.get('recommendations', []))
                except Exception as e:
                    logger.warning(f"AI enhancement failed: {e}")
            
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

    def _get_ai_insights(self, cv_data: CVData, section_scores: List[SectionScore]) -> Optional[Dict[str, List[str]]]:
        """Get AI-powered insights about the CV."""
        try:
            # Prepare CV summary for AI analysis
            cv_summary = self._prepare_cv_summary(cv_data, section_scores)

            # Create AI prompt
            prompt = f"""
Analyze this CV and provide insights:

{cv_summary}

Job Requirements:
- Required skills: {', '.join(self.evaluation_criteria.required_skills)}
- Preferred skills: {', '.join(self.evaluation_criteria.preferred_skills)}
- Minimum experience: {self.evaluation_criteria.min_experience_years} years

Please provide:
1. Top 3 strengths of this candidate
2. Top 3 areas for improvement
3. Top 3 specific recommendations

Format your response as JSON:
{{
    "strengths": ["strength1", "strength2", "strength3"],
    "weaknesses": ["weakness1", "weakness2", "weakness3"],
    "recommendations": ["rec1", "rec2", "rec3"]
}}
"""

            # Get AI response
            ai_response = get_ai_response(prompt, max_tokens=800)

            # Try to parse JSON response
            import json
            try:
                insights = json.loads(ai_response)
                return insights
            except json.JSONDecodeError:
                # If JSON parsing fails, extract insights from text
                return self._extract_insights_from_text(ai_response)

        except Exception as e:
            logger.error(f"AI insights generation failed: {e}")
            return None

    def _prepare_cv_summary(self, cv_data: CVData, section_scores: List[SectionScore]) -> str:
        """Prepare a concise CV summary for AI analysis."""
        summary_parts = []

        # Personal info
        if cv_data.personal_info.name:
            summary_parts.append(f"Candidate: {cv_data.personal_info.name}")

        # Skills
        if cv_data.skills:
            skills_text = ", ".join([skill.name for skill in cv_data.skills[:10]])
            summary_parts.append(f"Skills: {skills_text}")

        # Experience
        if cv_data.work_experience:
            exp_years = sum([exp.duration_months or 0 for exp in cv_data.work_experience]) / 12
            summary_parts.append(f"Total Experience: {exp_years:.1f} years")

            recent_exp = cv_data.work_experience[0] if cv_data.work_experience else None
            if recent_exp:
                summary_parts.append(f"Current Role: {recent_exp.position} at {recent_exp.company}")

        # Education
        if cv_data.education:
            highest_edu = cv_data.education[0] if cv_data.education else None
            if highest_edu:
                summary_parts.append(f"Education: {highest_edu.degree} from {highest_edu.institution}")

        # Section scores
        scores_text = ", ".join([f"{score.section}: {score.score:.1f}/{score.max_score:.1f}"
                                for score in section_scores])
        summary_parts.append(f"Scores: {scores_text}")

        return "\n".join(summary_parts)

    def _extract_insights_from_text(self, text: str) -> Dict[str, List[str]]:
        """Extract insights from unstructured AI response text."""
        insights = {"strengths": [], "weaknesses": [], "recommendations": []}

        lines = text.split('\n')
        current_section = None

        for line in lines:
            line = line.strip()
            if not line:
                continue

            # Detect section headers
            if any(word in line.lower() for word in ['strength', 'positive', 'good']):
                current_section = 'strengths'
            elif any(word in line.lower() for word in ['weakness', 'improvement', 'lacking']):
                current_section = 'weaknesses'
            elif any(word in line.lower() for word in ['recommendation', 'suggest', 'should']):
                current_section = 'recommendations'
            elif line.startswith(('-', '•', '*', '1.', '2.', '3.')) and current_section:
                # Extract bullet point
                insight = line.lstrip('-•*123. ').strip()
                if insight and len(insight) > 10:  # Minimum length filter
                    insights[current_section].append(insight)

        # Limit each section to 3 items
        for section in insights:
            insights[section] = insights[section][:3]

        return insights

    def chat_about_cv(self, cv_data: CVData, user_question: str) -> str:
        """Chat interface for asking questions about a CV."""
        if not self.ai_model_available:
            return self._get_fallback_chat_response(user_question)

        try:
            # Prepare CV context
            cv_context = self._prepare_cv_summary(cv_data, [])

            # Create chat prompt
            prompt = f"""
You are a CV evaluation expert. Here's information about a candidate:

{cv_context}

User question: {user_question}

Please provide a helpful, professional response about this candidate based on the CV information.
"""

            response = get_ai_response(prompt, max_tokens=600)
            return response

        except Exception as e:
            logger.error(f"Chat response generation failed: {e}")
            return f"I apologize, but I encountered an error while analyzing the CV. Error: {e}"

    def _get_fallback_chat_response(self, user_question: str) -> str:
        """Provide fallback response when AI is not available."""
        question_lower = user_question.lower()

        if any(word in question_lower for word in ['skill', 'technical', 'programming']):
            return "I can help analyze the candidate's skills based on the CV content. The system extracts and categorizes technical skills, soft skills, and experience levels from the CV text."

        elif any(word in question_lower for word in ['experience', 'work', 'job']):
            return "I can evaluate the candidate's work experience including duration, relevance to the position, and career progression based on the information in their CV."

        elif any(word in question_lower for word in ['education', 'degree', 'university']):
            return "I can assess the candidate's educational background including degrees, institutions, and how well they match the job requirements."

        elif any(word in question_lower for word in ['score', 'rating', 'evaluation']):
            return "The system provides detailed scoring across multiple categories including skills, experience, education, and additional factors, with customizable weights for each section."

        else:
            return "I can help you evaluate this CV across multiple dimensions. You can ask about the candidate's skills, experience, education, or overall fit for the position. What specific aspect would you like to know about?"
    
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
