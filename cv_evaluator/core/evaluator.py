"""
Main CV evaluator that coordinates all components.
"""

import logging
from pathlib import Path
from typing import Optional, Dict, Any
from ..core.models import CVAnalysisResult, EvaluationCriteria
from ..pdf.extractor import PDFExtractor
from ..pdf.parser import CVParser
from ..ai.analyzer import CVAnalyzer
from ..reports.generator import ReportGenerator
from ..core.criteria_loader import criteria_manager
from ..utils.exceptions import CVEvaluatorError
from ..utils.logging_config import setup_logging

logger = logging.getLogger(__name__)


class CVEvaluator:
    """Main CV evaluator that coordinates all evaluation components."""
    
    def __init__(self, 
                 evaluation_criteria: Optional[EvaluationCriteria] = None,
                 criteria_name: str = "default",
                 job_template: Optional[str] = None):
        """
        Initialize CV evaluator.
        
        Args:
            evaluation_criteria: Custom evaluation criteria
            criteria_name: Name of criteria configuration to load
            job_template: Job template to apply
        """
        # Setup logging
        setup_logging()
        
        # Load evaluation criteria
        if evaluation_criteria:
            self.evaluation_criteria = evaluation_criteria
        else:
            self.evaluation_criteria = criteria_manager.get_criteria(
                criteria_name, job_template
            )
        
        # Initialize components
        self.pdf_extractor = PDFExtractor()
        self.cv_parser = CVParser()
        self.analyzer = CVAnalyzer(self.evaluation_criteria)
        self.report_generator = ReportGenerator()
        
        logger.info("CV Evaluator initialized successfully")
    
    def evaluate_cv(self, cv_path: str) -> CVAnalysisResult:
        """
        Evaluate a single CV file.
        
        Args:
            cv_path: Path to the CV PDF file
            
        Returns:
            Complete CV analysis result
        """
        logger.info(f"Starting CV evaluation: {cv_path}")
        
        try:
            # Step 1: Extract text from PDF
            logger.info("Extracting text from PDF...")
            extraction_result = self.pdf_extractor.extract_text(cv_path)
            
            if not extraction_result.get('text', '').strip():
                raise CVEvaluatorError("No text could be extracted from the PDF")
            
            # Step 2: Parse CV structure
            logger.info("Parsing CV structure...")
            cv_data = self.cv_parser.parse_cv(extraction_result['text'])
            
            # Add extraction metadata
            cv_data.extraction_confidence = extraction_result.get('confidence', 0.0)
            
            # Step 3: Analyze CV with AI
            logger.info("Analyzing CV with AI...")
            analysis_result = self.analyzer.analyze_cv(cv_data)
            
            logger.info(f"CV evaluation completed. Score: {analysis_result.overall_score:.1f}")
            return analysis_result
            
        except Exception as e:
            logger.error(f"CV evaluation failed: {e}")
            raise CVEvaluatorError(f"Failed to evaluate CV: {e}")
    
    def generate_report(self,
                       analysis_result: CVAnalysisResult,
                       output_path: str,
                       format: str = "pdf",
                       template: str = "default",
                       **kwargs) -> str:
        """
        Generate evaluation report.
        
        Args:
            analysis_result: CV analysis result
            output_path: Path for output file
            format: Output format ('pdf', 'word', 'html')
            template: Template name
            **kwargs: Additional options
            
        Returns:
            Path to generated report
        """
        logger.info(f"Generating {format.upper()} report: {output_path}")
        
        try:
            report_path = self.report_generator.generate_report(
                analysis_result, output_path, format, template, **kwargs
            )
            
            logger.info(f"Report generated successfully: {report_path}")
            return report_path
            
        except Exception as e:
            logger.error(f"Report generation failed: {e}")
            raise CVEvaluatorError(f"Failed to generate report: {e}")
    
    def evaluate_and_report(self,
                           cv_path: str,
                           output_path: str,
                           format: str = "pdf",
                           template: str = "default",
                           **kwargs) -> Dict[str, Any]:
        """
        Evaluate CV and generate report in one step.
        
        Args:
            cv_path: Path to CV file
            output_path: Path for report output
            format: Report format
            template: Report template
            **kwargs: Additional options
            
        Returns:
            Dictionary with evaluation results and report path
        """
        # Evaluate CV
        analysis_result = self.evaluate_cv(cv_path)
        
        # Generate report
        report_path = self.generate_report(
            analysis_result, output_path, format, template, **kwargs
        )
        
        return {
            'analysis_result': analysis_result,
            'report_path': report_path,
            'overall_score': analysis_result.overall_score,
            'fit_percentage': analysis_result.fit_percentage,
            'candidate_name': analysis_result.cv_data.personal_info.name
        }
    
    def validate_cv_file(self, cv_path: str) -> bool:
        """
        Validate if a file is a readable CV.
        
        Args:
            cv_path: Path to CV file
            
        Returns:
            True if valid, False otherwise
        """
        try:
            return self.pdf_extractor.validate_pdf(cv_path)
        except Exception:
            return False
    
    def get_evaluation_criteria(self) -> EvaluationCriteria:
        """Get current evaluation criteria."""
        return self.evaluation_criteria
    
    def update_evaluation_criteria(self, 
                                 criteria: Optional[EvaluationCriteria] = None,
                                 criteria_name: Optional[str] = None,
                                 job_template: Optional[str] = None):
        """
        Update evaluation criteria.
        
        Args:
            criteria: Custom evaluation criteria
            criteria_name: Name of criteria configuration
            job_template: Job template to apply
        """
        if criteria:
            self.evaluation_criteria = criteria
        elif criteria_name:
            self.evaluation_criteria = criteria_manager.get_criteria(
                criteria_name, job_template
            )
        
        # Update analyzer with new criteria
        self.analyzer = CVAnalyzer(self.evaluation_criteria)
        
        logger.info("Evaluation criteria updated")
    
    def get_available_job_templates(self) -> list[str]:
        """Get list of available job templates."""
        return criteria_manager.list_job_templates()
    
    def get_available_criteria(self) -> list[str]:
        """Get list of available criteria configurations."""
        return criteria_manager.list_available_criteria()


def create_evaluator(criteria_name: str = "default", 
                    job_template: Optional[str] = None) -> CVEvaluator:
    """
    Factory function to create CV evaluator with specific configuration.
    
    Args:
        criteria_name: Name of criteria configuration
        job_template: Job template to apply
        
    Returns:
        Configured CV evaluator instance
    """
    return CVEvaluator(criteria_name=criteria_name, job_template=job_template)
