"""
CV parsing utilities to extract structured information from text.
"""

import re
import logging
from typing import List, Dict, Optional, Tuple
from datetime import datetime
from ..core.models import (
    CVData, PersonalInfo, Skill, Education, WorkExperience,
    SkillLevel, EducationLevel
)

logger = logging.getLogger(__name__)


class CVParser:
    """Parse CV text and extract structured information."""
    
    def __init__(self):
        self.email_pattern = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b')
        self.phone_pattern = re.compile(r'(\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}')
        self.linkedin_pattern = re.compile(r'linkedin\.com/in/[\w-]+', re.IGNORECASE)
        self.github_pattern = re.compile(r'github\.com/[\w-]+', re.IGNORECASE)
        
        # Common section headers
        self.section_patterns = {
            'experience': re.compile(r'(work\s+)?experience|employment|professional\s+experience', re.IGNORECASE),
            'education': re.compile(r'education|academic|qualifications', re.IGNORECASE),
            'skills': re.compile(r'skills|competencies|technical\s+skills', re.IGNORECASE),
            'projects': re.compile(r'projects|portfolio', re.IGNORECASE),
            'certifications': re.compile(r'certifications?|certificates?', re.IGNORECASE),
            'languages': re.compile(r'languages?', re.IGNORECASE)
        }
        
        # Common skill categories
        self.skill_categories = {
            'programming': ['python', 'java', 'javascript', 'c++', 'c#', 'php', 'ruby', 'go', 'rust'],
            'web': ['html', 'css', 'react', 'angular', 'vue', 'node.js', 'django', 'flask'],
            'database': ['sql', 'mysql', 'postgresql', 'mongodb', 'redis', 'oracle'],
            'cloud': ['aws', 'azure', 'gcp', 'docker', 'kubernetes'],
            'tools': ['git', 'jenkins', 'jira', 'confluence', 'slack']
        }
    
    def parse_cv(self, text: str) -> CVData:
        """
        Parse CV text and extract structured information.
        
        Args:
            text: Raw CV text
            
        Returns:
            CVData object with extracted information
        """
        logger.info("Starting CV parsing")
        
        # Extract personal information
        personal_info = self._extract_personal_info(text)
        
        # Split text into sections
        sections = self._split_into_sections(text)
        
        # Extract structured data from sections
        skills = self._extract_skills(sections.get('skills', ''))
        education = self._extract_education(sections.get('education', ''))
        work_experience = self._extract_work_experience(sections.get('experience', ''))
        languages = self._extract_languages(sections.get('languages', ''))
        certifications = self._extract_certifications(sections.get('certifications', ''))
        projects = self._extract_projects(sections.get('projects', ''))
        
        # Calculate extraction confidence
        confidence = self._calculate_confidence(personal_info, skills, education, work_experience)
        
        return CVData(
            personal_info=personal_info,
            skills=skills,
            education=education,
            work_experience=work_experience,
            languages=languages,
            certifications=certifications,
            projects=projects,
            raw_text=text,
            extraction_confidence=confidence
        )
    
    def _extract_personal_info(self, text: str) -> PersonalInfo:
        """Extract personal information from CV text."""
        # Extract email
        email_match = self.email_pattern.search(text)
        email = email_match.group() if email_match else None
        
        # Extract phone
        phone_match = self.phone_pattern.search(text)
        phone = phone_match.group() if phone_match else None
        
        # Extract LinkedIn
        linkedin_match = self.linkedin_pattern.search(text)
        linkedin = linkedin_match.group() if linkedin_match else None
        
        # Extract GitHub
        github_match = self.github_pattern.search(text)
        github = github_match.group() if github_match else None
        
        # Extract name (first few lines, excluding email/phone)
        lines = text.split('\n')[:5]
        name = None
        for line in lines:
            line = line.strip()
            if line and not self.email_pattern.search(line) and not self.phone_pattern.search(line):
                if len(line.split()) >= 2 and len(line) < 50:
                    name = line
                    break
        
        return PersonalInfo(
            name=name,
            email=email,
            phone=phone,
            linkedin=linkedin,
            github=github
        )
    
    def _split_into_sections(self, text: str) -> Dict[str, str]:
        """Split CV text into sections based on headers."""
        sections = {}
        current_section = 'general'
        current_content = []
        
        lines = text.split('\n')
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Check if line is a section header
            section_found = None
            for section_name, pattern in self.section_patterns.items():
                if pattern.search(line):
                    section_found = section_name
                    break
            
            if section_found:
                # Save previous section
                if current_content:
                    sections[current_section] = '\n'.join(current_content)
                
                # Start new section
                current_section = section_found
                current_content = []
            else:
                current_content.append(line)
        
        # Save last section
        if current_content:
            sections[current_section] = '\n'.join(current_content)
        
        return sections
    
    def _extract_skills(self, skills_text: str) -> List[Skill]:
        """Extract skills from skills section."""
        skills = []
        
        # Combine skills text with general text for broader skill detection
        text_lower = skills_text.lower()
        
        # Extract skills by category
        for category, skill_list in self.skill_categories.items():
            for skill_name in skill_list:
                if skill_name.lower() in text_lower:
                    skills.append(Skill(
                        name=skill_name,
                        category=category,
                        confidence=0.8
                    ))
        
        # Extract additional skills using patterns
        skill_patterns = [
            r'(\w+(?:\.\w+)*)\s*(?:[-–]\s*)?(?:proficient|experienced|expert|advanced|intermediate|beginner)',
            r'(?:proficient|experienced|expert|advanced|intermediate|beginner)\s+(?:in\s+)?(\w+(?:\.\w+)*)',
            r'(\w+(?:\s+\w+)*)\s*(?:\([^)]*\))?(?:\s*[-–]\s*\d+\s*years?)?'
        ]
        
        for pattern in skill_patterns:
            matches = re.finditer(pattern, skills_text, re.IGNORECASE)
            for match in matches:
                skill_name = match.group(1).strip()
                if len(skill_name) > 2 and skill_name.lower() not in [s.name.lower() for s in skills]:
                    skills.append(Skill(
                        name=skill_name,
                        confidence=0.6
                    ))
        
        return skills[:20]  # Limit to top 20 skills
    
    def _extract_education(self, education_text: str) -> List[Education]:
        """Extract education information."""
        education_list = []
        
        # Pattern for education entries
        education_pattern = re.compile(
            r'(?P<degree>bachelor|master|phd|doctorate|diploma|certificate|b\.?s\.?|m\.?s\.?|m\.?a\.?|b\.?a\.?)'
            r'.*?(?P<field>in\s+[\w\s]+)?.*?(?P<institution>university|college|institute|school).*?'
            r'(?P<year>\d{4})?',
            re.IGNORECASE | re.DOTALL
        )
        
        matches = education_pattern.finditer(education_text)
        for match in matches:
            degree = match.group('degree')
            field = match.group('field')
            institution = match.group('institution')
            year = match.group('year')
            
            education_list.append(Education(
                degree=degree,
                field_of_study=field.replace('in ', '') if field else None,
                institution=institution,
                graduation_year=int(year) if year else None
            ))
        
        return education_list
    
    def _extract_work_experience(self, experience_text: str) -> List[WorkExperience]:
        """Extract work experience information."""
        experience_list = []
        
        # Split by common separators
        entries = re.split(r'\n\s*\n|\n(?=[A-Z][^a-z]*(?:Engineer|Developer|Manager|Analyst))', experience_text)
        
        for entry in entries:
            if len(entry.strip()) < 20:  # Skip very short entries
                continue
            
            lines = [line.strip() for line in entry.split('\n') if line.strip()]
            if not lines:
                continue
            
            # First line usually contains position and company
            first_line = lines[0]
            
            # Extract position and company
            position_company_match = re.search(r'(.+?)\s+(?:at|@|-)\s+(.+)', first_line)
            if position_company_match:
                position = position_company_match.group(1).strip()
                company = position_company_match.group(2).strip()
            else:
                position = first_line
                company = "Unknown"
            
            # Extract dates
            date_pattern = re.compile(r'(\d{4})\s*[-–]\s*(\d{4}|present)', re.IGNORECASE)
            date_match = date_pattern.search(entry)
            start_date = date_match.group(1) if date_match else None
            end_date = date_match.group(2) if date_match else None
            
            # Extract description
            description = '\n'.join(lines[1:]) if len(lines) > 1 else ""
            
            experience_list.append(WorkExperience(
                position=position,
                company=company,
                start_date=start_date,
                end_date=end_date,
                description=description
            ))
        
        return experience_list
    
    def _extract_languages(self, languages_text: str) -> List[str]:
        """Extract languages from text."""
        common_languages = [
            'english', 'french', 'spanish', 'german', 'italian', 'portuguese',
            'chinese', 'japanese', 'korean', 'arabic', 'russian', 'hindi'
        ]
        
        found_languages = []
        text_lower = languages_text.lower()
        
        for lang in common_languages:
            if lang in text_lower:
                found_languages.append(lang.capitalize())
        
        return found_languages
    
    def _extract_certifications(self, cert_text: str) -> List[str]:
        """Extract certifications from text."""
        # Simple extraction - split by lines and clean
        certifications = []
        lines = cert_text.split('\n')
        
        for line in lines:
            line = line.strip()
            if line and len(line) > 5:
                certifications.append(line)
        
        return certifications[:10]  # Limit to 10 certifications
    
    def _extract_projects(self, projects_text: str) -> List[str]:
        """Extract projects from text."""
        projects = []
        lines = projects_text.split('\n')
        
        for line in lines:
            line = line.strip()
            if line and len(line) > 10:
                projects.append(line)
        
        return projects[:10]  # Limit to 10 projects
    
    def _calculate_confidence(self, personal_info: PersonalInfo, skills: List[Skill], 
                            education: List[Education], work_experience: List[WorkExperience]) -> float:
        """Calculate overall extraction confidence."""
        confidence_factors = []
        
        # Personal info confidence
        personal_score = 0
        if personal_info.name:
            personal_score += 0.3
        if personal_info.email:
            personal_score += 0.3
        if personal_info.phone:
            personal_score += 0.2
        if personal_info.linkedin or personal_info.github:
            personal_score += 0.2
        confidence_factors.append(personal_score)
        
        # Skills confidence
        skills_score = min(len(skills) / 10, 1.0)  # Normalize to 0-1
        confidence_factors.append(skills_score)
        
        # Education confidence
        education_score = min(len(education) / 3, 1.0)  # Normalize to 0-1
        confidence_factors.append(education_score)
        
        # Experience confidence
        experience_score = min(len(work_experience) / 3, 1.0)  # Normalize to 0-1
        confidence_factors.append(experience_score)
        
        return sum(confidence_factors) / len(confidence_factors)
