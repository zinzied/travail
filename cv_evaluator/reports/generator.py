"""
Main report generator that coordinates different output formats.
"""

import logging
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime
from ..core.models import CVAnalysisResult
from .pdf_generator import PDFReportGenerator
from .word_generator import WordReportGenerator
from .chart_generator import ChartGenerator
from ..utils.exceptions import ReportGenerationError

logger = logging.getLogger(__name__)


class ReportGenerator:
    """Main report generator that supports multiple output formats."""
    
    def __init__(self, template_dir: str = "templates"):
        self.template_dir = Path(template_dir)
        self.pdf_generator = PDFReportGenerator(template_dir)
        self.word_generator = WordReportGenerator(template_dir)
        self.chart_generator = ChartGenerator()
    
    def generate_report(self, 
                       analysis_result: CVAnalysisResult,
                       output_path: str,
                       format: str = "pdf",
                       template: str = "default",
                       include_charts: bool = True,
                       **kwargs) -> str:
        """
        Generate evaluation report in specified format.
        
        Args:
            analysis_result: CV analysis result
            output_path: Path for output file
            format: Output format ('pdf', 'word', 'html')
            template: Template name to use
            include_charts: Whether to include charts
            **kwargs: Additional options for specific generators
            
        Returns:
            Path to generated report file
        """
        logger.info(f"Generating {format.upper()} report: {output_path}")
        
        try:
            # Prepare report data
            report_data = self._prepare_report_data(analysis_result, include_charts)
            
            # Generate report based on format
            if format.lower() == "pdf":
                output_file = self.pdf_generator.generate(
                    report_data, output_path, template, **kwargs
                )
            elif format.lower() in ["word", "docx"]:
                output_file = self.word_generator.generate(
                    report_data, output_path, template, **kwargs
                )
            elif format.lower() == "html":
                output_file = self._generate_html_report(
                    report_data, output_path, template, **kwargs
                )
            else:
                raise ReportGenerationError(f"Unsupported format: {format}")
            
            logger.info(f"Report generated successfully: {output_file}")
            return output_file
            
        except Exception as e:
            logger.error(f"Report generation failed: {e}")
            raise ReportGenerationError(f"Failed to generate report: {e}")
    
    def generate_batch_report(self,
                            analysis_results: list[CVAnalysisResult],
                            output_dir: str,
                            format: str = "pdf",
                            template: str = "batch",
                            **kwargs) -> list[str]:
        """
        Generate reports for multiple CV analyses.
        
        Args:
            analysis_results: List of CV analysis results
            output_dir: Directory for output files
            format: Output format
            template: Template name
            **kwargs: Additional options
            
        Returns:
            List of paths to generated report files
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        generated_files = []
        
        for i, result in enumerate(analysis_results):
            try:
                # Generate filename from candidate name or index
                candidate_name = result.cv_data.personal_info.name or f"candidate_{i+1}"
                safe_name = "".join(c for c in candidate_name if c.isalnum() or c in (' ', '-', '_')).rstrip()
                filename = f"{safe_name}_evaluation.{format}"
                output_path = output_dir / filename
                
                report_file = self.generate_report(
                    result, str(output_path), format, template, **kwargs
                )
                generated_files.append(report_file)
                
            except Exception as e:
                logger.error(f"Failed to generate report for candidate {i+1}: {e}")
                continue
        
        logger.info(f"Generated {len(generated_files)} batch reports")
        return generated_files
    
    def _prepare_report_data(self, analysis_result: CVAnalysisResult, 
                           include_charts: bool = True) -> Dict[str, Any]:
        """Prepare data structure for report generation."""
        cv_data = analysis_result.cv_data
        
        # Basic report data
        report_data = {
            'metadata': {
                'generated_at': datetime.now(),
                'candidate_name': cv_data.personal_info.name or "Unknown",
                'overall_score': analysis_result.overall_score,
                'fit_percentage': analysis_result.fit_percentage,
                'analysis_timestamp': analysis_result.analysis_timestamp
            },
            'personal_info': {
                'name': cv_data.personal_info.name,
                'email': cv_data.personal_info.email,
                'phone': cv_data.personal_info.phone,
                'location': cv_data.personal_info.location,
                'linkedin': cv_data.personal_info.linkedin,
                'github': cv_data.personal_info.github
            },
            'summary': {
                'overall_score': analysis_result.overall_score,
                'fit_percentage': analysis_result.fit_percentage,
                'strengths_count': len(analysis_result.strengths),
                'weaknesses_count': len(analysis_result.weaknesses),
                'recommendations_count': len(analysis_result.recommendations)
            },
            'section_scores': [
                {
                    'section': score.section.title(),
                    'score': score.score,
                    'max_score': score.max_score,
                    'percentage': (score.score / score.max_score) * 100,
                    'feedback': score.feedback,
                    'details': score.details
                }
                for score in analysis_result.section_scores
            ],
            'skills': [
                {
                    'name': skill.name,
                    'category': skill.category or 'General',
                    'level': skill.level.value if skill.level else 'Unknown',
                    'years_experience': skill.years_experience,
                    'confidence': skill.confidence
                }
                for skill in cv_data.skills
            ],
            'experience': [
                {
                    'company': exp.company,
                    'position': exp.position,
                    'start_date': exp.start_date,
                    'end_date': exp.end_date,
                    'duration_months': exp.duration_months,
                    'description': exp.description,
                    'technologies': exp.technologies or [],
                    'achievements': exp.achievements or []
                }
                for exp in cv_data.work_experience
            ],
            'education': [
                {
                    'institution': edu.institution,
                    'degree': edu.degree,
                    'field_of_study': edu.field_of_study,
                    'graduation_year': edu.graduation_year,
                    'level': edu.level.value if edu.level else 'Unknown'
                }
                for edu in cv_data.education
            ],
            'additional': {
                'languages': cv_data.languages,
                'certifications': cv_data.certifications,
                'projects': cv_data.projects
            },
            'analysis': {
                'strengths': analysis_result.strengths,
                'weaknesses': analysis_result.weaknesses,
                'recommendations': analysis_result.recommendations
            },
            'criteria': {
                'required_skills': analysis_result.evaluation_criteria.required_skills if analysis_result.evaluation_criteria else [],
                'preferred_skills': analysis_result.evaluation_criteria.preferred_skills if analysis_result.evaluation_criteria else [],
                'min_experience_years': analysis_result.evaluation_criteria.min_experience_years if analysis_result.evaluation_criteria else 0,
                'scoring_weights': analysis_result.evaluation_criteria.scoring_weights if analysis_result.evaluation_criteria else {}
            }
        }
        
        # Generate charts if requested
        if include_charts:
            report_data['charts'] = self._generate_charts(analysis_result)
        
        return report_data
    
    def _generate_charts(self, analysis_result: CVAnalysisResult) -> Dict[str, str]:
        """Generate charts for the report."""
        charts = {}
        
        try:
            # Score breakdown chart
            charts['score_breakdown'] = self.chart_generator.create_score_breakdown_chart(
                analysis_result.section_scores
            )
            
            # Skills radar chart
            if analysis_result.cv_data.skills:
                charts['skills_radar'] = self.chart_generator.create_skills_radar_chart(
                    analysis_result.cv_data.skills
                )
            
            # Experience timeline
            if analysis_result.cv_data.work_experience:
                charts['experience_timeline'] = self.chart_generator.create_experience_timeline(
                    analysis_result.cv_data.work_experience
                )
            
        except Exception as e:
            logger.warning(f"Failed to generate some charts: {e}")
        
        return charts
    
    def _generate_html_report(self, report_data: Dict[str, Any], 
                            output_path: str, template: str, **kwargs) -> str:
        """Generate HTML report (basic implementation)."""
        from jinja2 import Environment, FileSystemLoader
        
        # Setup Jinja2 environment
        template_path = self.template_dir / "html"
        if not template_path.exists():
            template_path = self.template_dir
        
        env = Environment(loader=FileSystemLoader(str(template_path)))
        
        # Load template
        template_file = f"{template}.html"
        if not (template_path / template_file).exists():
            template_file = "default.html"
        
        try:
            template_obj = env.get_template(template_file)
        except Exception:
            # Fallback to basic template
            template_obj = env.from_string(self._get_basic_html_template())
        
        # Render template
        html_content = template_obj.render(**report_data)
        
        # Write to file
        output_path = Path(output_path)
        if output_path.suffix.lower() != '.html':
            output_path = output_path.with_suffix('.html')
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        return str(output_path)
    
    def _get_basic_html_template(self) -> str:
        """Get basic HTML template as fallback."""
        return """
<!DOCTYPE html>
<html>
<head>
    <title>CV Evaluation Report - {{ metadata.candidate_name }}</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 40px; }
        .header { border-bottom: 2px solid #333; padding-bottom: 20px; }
        .section { margin: 30px 0; }
        .score { font-size: 24px; font-weight: bold; color: #2c5aa0; }
        .chart { margin: 20px 0; text-align: center; }
        table { width: 100%; border-collapse: collapse; margin: 20px 0; }
        th, td { border: 1px solid #ddd; padding: 12px; text-align: left; }
        th { background-color: #f2f2f2; }
    </style>
</head>
<body>
    <div class="header">
        <h1>CV Evaluation Report</h1>
        <h2>{{ metadata.candidate_name }}</h2>
        <p>Generated on: {{ metadata.generated_at.strftime('%Y-%m-%d %H:%M:%S') }}</p>
    </div>
    
    <div class="section">
        <h3>Overall Score</h3>
        <div class="score">{{ "%.1f"|format(summary.overall_score) }}/100</div>
        <p>Fit Percentage: {{ "%.1f"|format(summary.fit_percentage) }}%</p>
    </div>
    
    <div class="section">
        <h3>Section Scores</h3>
        <table>
            <tr><th>Section</th><th>Score</th><th>Percentage</th><th>Feedback</th></tr>
            {% for score in section_scores %}
            <tr>
                <td>{{ score.section }}</td>
                <td>{{ "%.1f"|format(score.score) }}/{{ "%.1f"|format(score.max_score) }}</td>
                <td>{{ "%.1f"|format(score.percentage) }}%</td>
                <td>{{ score.feedback }}</td>
            </tr>
            {% endfor %}
        </table>
    </div>
    
    <div class="section">
        <h3>Strengths</h3>
        <ul>
            {% for strength in analysis.strengths %}
            <li>{{ strength }}</li>
            {% endfor %}
        </ul>
    </div>
    
    <div class="section">
        <h3>Areas for Improvement</h3>
        <ul>
            {% for weakness in analysis.weaknesses %}
            <li>{{ weakness }}</li>
            {% endfor %}
        </ul>
    </div>
    
    <div class="section">
        <h3>Recommendations</h3>
        <ul>
            {% for recommendation in analysis.recommendations %}
            <li>{{ recommendation }}</li>
            {% endfor %}
        </ul>
    </div>
</body>
</html>
        """
