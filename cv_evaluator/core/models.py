"""
Core data models for CV evaluation system.
"""

from typing import List, Dict, Optional, Any
from datetime import datetime
from pydantic import BaseModel, Field
from enum import Enum


class SkillLevel(str, Enum):
    """Skill proficiency levels."""
    BEGINNER = "beginner"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"
    EXPERT = "expert"


class EducationLevel(str, Enum):
    """Education levels."""
    HIGH_SCHOOL = "high_school"
    BACHELOR = "bachelor"
    MASTER = "master"
    PHD = "phd"
    CERTIFICATION = "certification"


class ExperienceLevel(str, Enum):
    """Experience levels."""
    ENTRY = "entry"
    JUNIOR = "junior"
    MID = "mid"
    SENIOR = "senior"
    LEAD = "lead"
    EXECUTIVE = "executive"


class Skill(BaseModel):
    """Represents a skill extracted from CV."""
    name: str
    level: Optional[SkillLevel] = None
    years_experience: Optional[int] = None
    category: Optional[str] = None
    confidence: float = Field(ge=0.0, le=1.0, default=0.0)


class Education(BaseModel):
    """Represents education information."""
    institution: str
    degree: str
    field_of_study: Optional[str] = None
    level: Optional[EducationLevel] = None
    graduation_year: Optional[int] = None
    gpa: Optional[float] = None
    honors: Optional[List[str]] = None


class WorkExperience(BaseModel):
    """Represents work experience."""
    company: str
    position: str
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    duration_months: Optional[int] = None
    description: Optional[str] = None
    responsibilities: Optional[List[str]] = None
    achievements: Optional[List[str]] = None
    technologies: Optional[List[str]] = None


class PersonalInfo(BaseModel):
    """Personal information from CV."""
    name: Optional[str] = None
    email: Optional[str] = None
    phone: Optional[str] = None
    location: Optional[str] = None
    linkedin: Optional[str] = None
    github: Optional[str] = None
    website: Optional[str] = None


class CVData(BaseModel):
    """Structured CV data extracted from PDF."""
    personal_info: PersonalInfo
    skills: List[Skill] = []
    education: List[Education] = []
    work_experience: List[WorkExperience] = []
    languages: List[str] = []
    certifications: List[str] = []
    projects: List[str] = []
    raw_text: str = ""
    extraction_confidence: float = Field(ge=0.0, le=1.0, default=0.0)


class EvaluationCriteria(BaseModel):
    """Evaluation criteria configuration."""
    required_skills: List[str] = []
    preferred_skills: List[str] = []
    min_experience_years: int = 0
    required_education_level: Optional[EducationLevel] = None
    industry_keywords: List[str] = []
    scoring_weights: Dict[str, float] = {
        "skills": 0.4,
        "experience": 0.3,
        "education": 0.2,
        "additional": 0.1
    }
    max_score: int = 100


class SectionScore(BaseModel):
    """Score for a specific CV section."""
    section: str
    score: float = Field(ge=0.0, le=100.0)
    max_score: float = Field(ge=0.0, le=100.0)
    details: Dict[str, Any] = {}
    feedback: Optional[str] = None


class CVAnalysisResult(BaseModel):
    """Complete CV analysis result."""
    cv_data: CVData
    overall_score: float = Field(ge=0.0, le=100.0)
    section_scores: List[SectionScore] = []
    strengths: List[str] = []
    weaknesses: List[str] = []
    recommendations: List[str] = []
    fit_percentage: float = Field(ge=0.0, le=100.0, default=0.0)
    analysis_timestamp: datetime = Field(default_factory=datetime.now)
    evaluation_criteria: Optional[EvaluationCriteria] = None
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }
