"""
Tests for the main CV evaluator.
"""

import pytest
from unittest.mock import Mock, patch
from cv_evaluator.core.evaluator import CVEvaluator
from cv_evaluator.core.models import (
    CVData, PersonalInfo, Skill, CVAnalysisResult, 
    EvaluationCriteria, SectionScore
)
from cv_evaluator.utils.exceptions import CVEvaluatorError


class TestCVEvaluator:
    """Test CVEvaluator functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        # Create test evaluation criteria
        self.test_criteria = EvaluationCriteria(
            required_skills=["python", "sql"],
            preferred_skills=["machine learning"],
            min_experience_years=2,
            scoring_weights={
                "skills": 0.4,
                "experience": 0.3,
                "education": 0.2,
                "additional": 0.1
            }
        )
        
        self.evaluator = CVEvaluator(evaluation_criteria=self.test_criteria)
    
    def test_evaluator_initialization(self):
        """Test evaluator initialization."""
        assert self.evaluator.evaluation_criteria == self.test_criteria
        assert self.evaluator.pdf_extractor is not None
        assert self.evaluator.cv_parser is not None
        assert self.evaluator.analyzer is not None
        assert self.evaluator.report_generator is not None
    
    @patch('cv_evaluator.core.evaluator.CVEvaluator.validate_cv_file')
    def test_validate_cv_file(self, mock_validate):
        """Test CV file validation."""
        mock_validate.return_value = True
        
        result = self.evaluator.validate_cv_file("test.pdf")
        assert result is True
        
        mock_validate.return_value = False
        result = self.evaluator.validate_cv_file("invalid.pdf")
        assert result is False
    
    def test_get_evaluation_criteria(self):
        """Test getting evaluation criteria."""
        criteria = self.evaluator.get_evaluation_criteria()
        assert criteria == self.test_criteria
    
    def test_update_evaluation_criteria(self):
        """Test updating evaluation criteria."""
        new_criteria = EvaluationCriteria(
            required_skills=["java", "spring"],
            min_experience_years=5
        )
        
        self.evaluator.update_evaluation_criteria(criteria=new_criteria)
        assert self.evaluator.evaluation_criteria == new_criteria
    
    @patch('cv_evaluator.pdf.extractor.PDFExtractor.extract_text')
    @patch('cv_evaluator.pdf.parser.CVParser.parse_cv')
    @patch('cv_evaluator.ai.analyzer.CVAnalyzer.analyze_cv')
    def test_evaluate_cv_success(self, mock_analyze, mock_parse, mock_extract):
        """Test successful CV evaluation."""
        # Mock PDF extraction
        mock_extract.return_value = {
            'text': 'Sample CV text with Python and SQL skills',
            'confidence': 0.9
        }
        
        # Mock CV parsing
        mock_cv_data = CVData(
            personal_info=PersonalInfo(name="John Doe"),
            skills=[
                Skill(name="Python", confidence=0.9),
                Skill(name="SQL", confidence=0.8)
            ],
            extraction_confidence=0.9
        )
        mock_parse.return_value = mock_cv_data
        
        # Mock analysis
        mock_analysis_result = CVAnalysisResult(
            cv_data=mock_cv_data,
            overall_score=85.0,
            section_scores=[
                SectionScore(section="skills", score=90.0, max_score=100.0),
                SectionScore(section="experience", score=80.0, max_score=100.0)
            ],
            strengths=["Strong technical skills"],
            weaknesses=["Limited experience"],
            recommendations=["Gain more experience"],
            fit_percentage=85.0
        )
        mock_analyze.return_value = mock_analysis_result
        
        # Test evaluation
        result = self.evaluator.evaluate_cv("test.pdf")
        
        assert isinstance(result, CVAnalysisResult)
        assert result.overall_score == 85.0
        assert result.fit_percentage == 85.0
        assert len(result.section_scores) == 2
        
        # Verify mocks were called
        mock_extract.assert_called_once_with("test.pdf")
        mock_parse.assert_called_once()
        mock_analyze.assert_called_once()
    
    @patch('cv_evaluator.pdf.extractor.PDFExtractor.extract_text')
    def test_evaluate_cv_extraction_failure(self, mock_extract):
        """Test CV evaluation with extraction failure."""
        mock_extract.side_effect = Exception("PDF extraction failed")
        
        with pytest.raises(CVEvaluatorError) as exc_info:
            self.evaluator.evaluate_cv("test.pdf")
        
        assert "Failed to evaluate CV" in str(exc_info.value)
    
    @patch('cv_evaluator.pdf.extractor.PDFExtractor.extract_text')
    def test_evaluate_cv_empty_text(self, mock_extract):
        """Test CV evaluation with empty extracted text."""
        mock_extract.return_value = {'text': '', 'confidence': 0.0}
        
        with pytest.raises(CVEvaluatorError) as exc_info:
            self.evaluator.evaluate_cv("test.pdf")
        
        assert "No text could be extracted" in str(exc_info.value)
    
    @patch('cv_evaluator.core.evaluator.CVEvaluator.evaluate_cv')
    @patch('cv_evaluator.reports.generator.ReportGenerator.generate_report')
    def test_evaluate_and_report(self, mock_generate_report, mock_evaluate):
        """Test evaluate and report functionality."""
        # Mock evaluation result
        mock_analysis_result = CVAnalysisResult(
            cv_data=CVData(personal_info=PersonalInfo(name="John Doe")),
            overall_score=85.0,
            section_scores=[],
            strengths=[],
            weaknesses=[],
            recommendations=[],
            fit_percentage=85.0
        )
        mock_evaluate.return_value = mock_analysis_result
        
        # Mock report generation
        mock_generate_report.return_value = "report.pdf"
        
        # Test evaluate and report
        result = self.evaluator.evaluate_and_report(
            "test.pdf", "output.pdf", format="pdf"
        )
        
        assert result['overall_score'] == 85.0
        assert result['fit_percentage'] == 85.0
        assert result['candidate_name'] == "John Doe"
        assert result['report_path'] == "report.pdf"
        assert result['analysis_result'] == mock_analysis_result
        
        # Verify mocks were called
        mock_evaluate.assert_called_once_with("test.pdf")
        mock_generate_report.assert_called_once()
    
    def test_get_available_job_templates(self):
        """Test getting available job templates."""
        # This test depends on the criteria manager having templates
        templates = self.evaluator.get_available_job_templates()
        assert isinstance(templates, list)
    
    def test_get_available_criteria(self):
        """Test getting available criteria configurations."""
        criteria_list = self.evaluator.get_available_criteria()
        assert isinstance(criteria_list, list)


class TestEvaluatorFactory:
    """Test evaluator factory functions."""
    
    def test_create_evaluator_default(self):
        """Test creating evaluator with default settings."""
        from cv_evaluator.core.evaluator import create_evaluator
        
        evaluator = create_evaluator()
        assert isinstance(evaluator, CVEvaluator)
        assert evaluator.evaluation_criteria is not None
    
    def test_create_evaluator_with_template(self):
        """Test creating evaluator with job template."""
        from cv_evaluator.core.evaluator import create_evaluator
        
        evaluator = create_evaluator(job_template="software_engineer")
        assert isinstance(evaluator, CVEvaluator)
        assert evaluator.evaluation_criteria is not None


if __name__ == "__main__":
    pytest.main([__file__])
