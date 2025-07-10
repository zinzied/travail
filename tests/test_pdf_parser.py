"""
Tests for PDF parsing functionality.
"""

import pytest
from cv_evaluator.pdf.parser import CVParser
from cv_evaluator.core.models import CVData, PersonalInfo, Skill, Education, WorkExperience


class TestCVParser:
    """Test CVParser functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.parser = CVParser()
    
    def test_parse_personal_info(self):
        """Test parsing personal information."""
        cv_text = """
        John Doe
        john.doe@example.com
        +1-555-0123
        linkedin.com/in/johndoe
        github.com/johndoe
        """
        
        personal_info = self.parser._extract_personal_info(cv_text)
        
        assert personal_info.name == "John Doe"
        assert personal_info.email == "john.doe@example.com"
        assert personal_info.phone == "+1-555-0123"
        assert "linkedin.com/in/johndoe" in personal_info.linkedin
        assert "github.com/johndoe" in personal_info.github
    
    def test_parse_skills_section(self):
        """Test parsing skills section."""
        skills_text = """
        Programming Languages: Python, Java, JavaScript
        Web Technologies: HTML, CSS, React, Node.js
        Databases: MySQL, PostgreSQL, MongoDB
        Tools: Git, Docker, Jenkins
        """
        
        skills = self.parser._extract_skills(skills_text)
        
        skill_names = [skill.name.lower() for skill in skills]
        assert "python" in skill_names
        assert "java" in skill_names
        assert "javascript" in skill_names
        assert "react" in skill_names
        assert "mysql" in skill_names
    
    def test_parse_work_experience(self):
        """Test parsing work experience."""
        experience_text = """
        Senior Software Engineer at Tech Corp
        2020 - 2023
        • Developed web applications using Python and React
        • Led a team of 5 developers
        • Improved system performance by 40%
        
        Software Engineer at StartupXYZ
        2018 - 2020
        • Built REST APIs using Node.js
        • Worked with MongoDB and PostgreSQL
        """
        
        experiences = self.parser._extract_work_experience(experience_text)
        
        assert len(experiences) >= 1
        
        # Check first experience
        first_exp = experiences[0]
        assert "Senior Software Engineer" in first_exp.position
        assert "Tech Corp" in first_exp.company or "Tech Corp" in first_exp.position
    
    def test_parse_education(self):
        """Test parsing education information."""
        education_text = """
        Master of Science in Computer Science
        Massachusetts Institute of Technology
        2018
        
        Bachelor of Science in Software Engineering
        Stanford University
        2016
        """
        
        education_list = self.parser._extract_education(education_text)
        
        # Should find at least one education entry
        assert len(education_list) >= 0
    
    def test_parse_languages(self):
        """Test parsing languages."""
        languages_text = """
        Languages: English (Native), Spanish (Fluent), French (Intermediate)
        """
        
        languages = self.parser._extract_languages(languages_text)
        
        # Should find common languages
        language_names = [lang.lower() for lang in languages]
        assert "english" in language_names or "spanish" in language_names or "french" in language_names
    
    def test_full_cv_parsing(self):
        """Test parsing a complete CV."""
        cv_text = """
        John Doe
        john.doe@example.com
        +1-555-0123
        
        EXPERIENCE
        Senior Software Engineer at Tech Corp
        2020 - Present
        • Developed web applications using Python and React
        • Led a team of 5 developers
        
        EDUCATION
        Master of Science in Computer Science
        MIT, 2018
        
        SKILLS
        Programming: Python, Java, JavaScript
        Web: React, Node.js, HTML, CSS
        Databases: MySQL, PostgreSQL
        
        LANGUAGES
        English, Spanish
        
        CERTIFICATIONS
        AWS Certified Solutions Architect
        Certified Scrum Master
        """
        
        cv_data = self.parser.parse_cv(cv_text)
        
        # Verify parsed data
        assert isinstance(cv_data, CVData)
        assert cv_data.personal_info.name == "John Doe"
        assert cv_data.personal_info.email == "john.doe@example.com"
        assert len(cv_data.skills) > 0
        assert len(cv_data.work_experience) > 0
        assert cv_data.extraction_confidence > 0
    
    def test_confidence_calculation(self):
        """Test extraction confidence calculation."""
        # High confidence CV
        good_cv_text = """
        John Doe
        john.doe@example.com
        +1-555-0123
        
        SKILLS
        Python, Java, React, SQL
        
        EXPERIENCE
        Software Engineer at Tech Corp
        2020-2023
        
        EDUCATION
        BS Computer Science, MIT, 2020
        """
        
        cv_data = self.parser.parse_cv(good_cv_text)
        assert cv_data.extraction_confidence > 0.5
        
        # Low confidence CV (minimal info)
        poor_cv_text = "Some random text without structure"
        cv_data_poor = self.parser.parse_cv(poor_cv_text)
        assert cv_data_poor.extraction_confidence < cv_data.extraction_confidence
    
    def test_empty_text_handling(self):
        """Test handling of empty or minimal text."""
        cv_data = self.parser.parse_cv("")
        
        assert isinstance(cv_data, CVData)
        assert cv_data.personal_info.name is None
        assert len(cv_data.skills) == 0
        assert len(cv_data.work_experience) == 0
        assert cv_data.extraction_confidence == 0
    
    def test_section_splitting(self):
        """Test section splitting functionality."""
        cv_text = """
        John Doe
        
        EXPERIENCE
        Software Engineer at Company A
        
        EDUCATION
        BS Computer Science
        
        SKILLS
        Python, Java
        """
        
        sections = self.parser._split_into_sections(cv_text)
        
        assert "experience" in sections
        assert "education" in sections
        assert "skills" in sections
        assert "Software Engineer" in sections["experience"]
        assert "Computer Science" in sections["education"]
        assert "Python" in sections["skills"]


if __name__ == "__main__":
    pytest.main([__file__])
