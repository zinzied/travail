"""
Tests for core data models.
"""

import pytest
from datetime import datetime
from cv_evaluator.core.models import (
    CVData, PersonalInfo, Skill, Education, WorkExperience,
    EvaluationCriteria, CVAnalysisResult, SectionScore,
    SkillLevel, EducationLevel
)


class TestPersonalInfo:
    """Test PersonalInfo model."""
    
    def test_personal_info_creation(self):
        """Test creating PersonalInfo instance."""
        info = PersonalInfo(
            name="John Doe",
            email="john.doe@example.com",
            phone="+1-555-0123",
            location="New York, NY"
        )
        
        assert info.name == "John Doe"
        assert info.email == "john.doe@example.com"
        assert info.phone == "+1-555-0123"
        assert info.location == "New York, NY"
    
    def test_personal_info_optional_fields(self):
        """Test PersonalInfo with optional fields."""
        info = PersonalInfo()
        
        assert info.name is None
        assert info.email is None
        assert info.phone is None
        assert info.location is None


class TestSkill:
    """Test Skill model."""
    
    def test_skill_creation(self):
        """Test creating Skill instance."""
        skill = Skill(
            name="Python",
            level=SkillLevel.ADVANCED,
            years_experience=5,
            category="programming",
            confidence=0.9
        )
        
        assert skill.name == "Python"
        assert skill.level == SkillLevel.ADVANCED
        assert skill.years_experience == 5
        assert skill.category == "programming"
        assert skill.confidence == 0.9
    
    def test_skill_confidence_validation(self):
        """Test skill confidence validation."""
        # Valid confidence
        skill = Skill(name="Python", confidence=0.5)
        assert skill.confidence == 0.5
        
        # Test boundary values
        skill_min = Skill(name="Python", confidence=0.0)
        assert skill_min.confidence == 0.0
        
        skill_max = Skill(name="Python", confidence=1.0)
        assert skill_max.confidence == 1.0


class TestEducation:
    """Test Education model."""
    
    def test_education_creation(self):
        """Test creating Education instance."""
        education = Education(
            institution="MIT",
            degree="Bachelor of Science",
            field_of_study="Computer Science",
            level=EducationLevel.BACHELOR,
            graduation_year=2020
        )
        
        assert education.institution == "MIT"
        assert education.degree == "Bachelor of Science"
        assert education.field_of_study == "Computer Science"
        assert education.level == EducationLevel.BACHELOR
        assert education.graduation_year == 2020


class TestWorkExperience:
    """Test WorkExperience model."""
    
    def test_work_experience_creation(self):
        """Test creating WorkExperience instance."""
        experience = WorkExperience(
            company="Tech Corp",
            position="Software Engineer",
            start_date="2020-01",
            end_date="2023-12",
            duration_months=36,
            description="Developed web applications"
        )
        
        assert experience.company == "Tech Corp"
        assert experience.position == "Software Engineer"
        assert experience.start_date == "2020-01"
        assert experience.end_date == "2023-12"
        assert experience.duration_months == 36
        assert experience.description == "Developed web applications"


class TestCVData:
    """Test CVData model."""
    
    def test_cv_data_creation(self):
        """Test creating CVData instance."""
        personal_info = PersonalInfo(name="John Doe")
        skills = [Skill(name="Python")]
        education = [Education(institution="MIT", degree="BS")]
        
        cv_data = CVData(
            personal_info=personal_info,
            skills=skills,
            education=education,
            raw_text="Sample CV text",
            extraction_confidence=0.8
        )
        
        assert cv_data.personal_info.name == "John Doe"
        assert len(cv_data.skills) == 1
        assert cv_data.skills[0].name == "Python"
        assert len(cv_data.education) == 1
        assert cv_data.raw_text == "Sample CV text"
        assert cv_data.extraction_confidence == 0.8


class TestEvaluationCriteria:
    """Test EvaluationCriteria model."""
    
    def test_evaluation_criteria_creation(self):
        """Test creating EvaluationCriteria instance."""
        criteria = EvaluationCriteria(
            required_skills=["Python", "SQL"],
            preferred_skills=["Machine Learning"],
            min_experience_years=3,
            scoring_weights={
                "skills": 0.4,
                "experience": 0.3,
                "education": 0.2,
                "additional": 0.1
            }
        )
        
        assert criteria.required_skills == ["Python", "SQL"]
        assert criteria.preferred_skills == ["Machine Learning"]
        assert criteria.min_experience_years == 3
        assert criteria.scoring_weights["skills"] == 0.4


class TestSectionScore:
    """Test SectionScore model."""
    
    def test_section_score_creation(self):
        """Test creating SectionScore instance."""
        score = SectionScore(
            section="skills",
            score=85.0,
            max_score=100.0,
            details={"skill_count": 10},
            feedback="Good skill diversity"
        )
        
        assert score.section == "skills"
        assert score.score == 85.0
        assert score.max_score == 100.0
        assert score.details["skill_count"] == 10
        assert score.feedback == "Good skill diversity"
    
    def test_section_score_validation(self):
        """Test SectionScore validation."""
        # Valid scores
        score = SectionScore(section="test", score=50.0, max_score=100.0)
        assert score.score == 50.0
        
        # Boundary values
        score_min = SectionScore(section="test", score=0.0, max_score=100.0)
        assert score_min.score == 0.0
        
        score_max = SectionScore(section="test", score=100.0, max_score=100.0)
        assert score_max.score == 100.0


class TestCVAnalysisResult:
    """Test CVAnalysisResult model."""
    
    def test_cv_analysis_result_creation(self):
        """Test creating CVAnalysisResult instance."""
        # Create sample data
        personal_info = PersonalInfo(name="John Doe")
        cv_data = CVData(personal_info=personal_info)
        
        section_scores = [
            SectionScore(section="skills", score=80.0, max_score=100.0),
            SectionScore(section="experience", score=70.0, max_score=100.0)
        ]
        
        result = CVAnalysisResult(
            cv_data=cv_data,
            overall_score=75.0,
            section_scores=section_scores,
            strengths=["Strong technical skills"],
            weaknesses=["Limited experience"],
            recommendations=["Gain more experience"],
            fit_percentage=80.0
        )
        
        assert result.cv_data.personal_info.name == "John Doe"
        assert result.overall_score == 75.0
        assert len(result.section_scores) == 2
        assert result.strengths == ["Strong technical skills"]
        assert result.weaknesses == ["Limited experience"]
        assert result.recommendations == ["Gain more experience"]
        assert result.fit_percentage == 80.0
        assert isinstance(result.analysis_timestamp, datetime)


if __name__ == "__main__":
    pytest.main([__file__])
