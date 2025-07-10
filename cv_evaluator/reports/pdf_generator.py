"""
PDF report generator using ReportLab.
"""

import logging
from pathlib import Path
from typing import Dict, Any, List
from datetime import datetime
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    PageBreak, Image, KeepTogether
)
from reportlab.graphics.shapes import Drawing
from reportlab.graphics.charts.barcharts import VerticalBarChart
from reportlab.graphics.charts.piecharts import Pie
from ..utils.exceptions import ReportGenerationError

logger = logging.getLogger(__name__)


class PDFReportGenerator:
    """Generate PDF evaluation reports using ReportLab."""
    
    def __init__(self, template_dir: str = "templates"):
        self.template_dir = Path(template_dir)
        self.styles = getSampleStyleSheet()
        self._setup_custom_styles()
    
    def generate(self, report_data: Dict[str, Any], output_path: str, 
                template: str = "default", **kwargs) -> str:
        """
        Generate PDF report.
        
        Args:
            report_data: Report data dictionary
            output_path: Output file path
            template: Template name (not used in this implementation)
            **kwargs: Additional options
            
        Returns:
            Path to generated PDF file
        """
        try:
            output_path = Path(output_path)
            if output_path.suffix.lower() != '.pdf':
                output_path = output_path.with_suffix('.pdf')
            
            # Create PDF document
            doc = SimpleDocTemplate(
                str(output_path),
                pagesize=A4,
                rightMargin=72,
                leftMargin=72,
                topMargin=72,
                bottomMargin=18
            )
            
            # Build content
            story = []
            self._add_header(story, report_data)
            self._add_summary(story, report_data)
            self._add_section_scores(story, report_data)
            self._add_detailed_analysis(story, report_data)
            self._add_skills_analysis(story, report_data)
            self._add_experience_analysis(story, report_data)
            self._add_education_analysis(story, report_data)
            self._add_recommendations(story, report_data)
            
            # Build PDF
            doc.build(story)
            
            logger.info(f"PDF report generated: {output_path}")
            return str(output_path)
            
        except Exception as e:
            logger.error(f"PDF generation failed: {e}")
            raise ReportGenerationError(f"Failed to generate PDF report: {e}")
    
    def _setup_custom_styles(self):
        """Setup custom paragraph styles."""
        self.styles.add(ParagraphStyle(
            name='CustomTitle',
            parent=self.styles['Heading1'],
            fontSize=24,
            spaceAfter=30,
            textColor=colors.HexColor('#2c5aa0'),
            alignment=1  # Center alignment
        ))
        
        self.styles.add(ParagraphStyle(
            name='SectionHeader',
            parent=self.styles['Heading2'],
            fontSize=16,
            spaceBefore=20,
            spaceAfter=12,
            textColor=colors.HexColor('#2c5aa0'),
            borderWidth=1,
            borderColor=colors.HexColor('#2c5aa0'),
            borderPadding=5
        ))
        
        self.styles.add(ParagraphStyle(
            name='ScoreText',
            parent=self.styles['Normal'],
            fontSize=18,
            textColor=colors.HexColor('#2c5aa0'),
            alignment=1,
            spaceAfter=10
        ))
    
    def _add_header(self, story: List, report_data: Dict[str, Any]):
        """Add report header."""
        metadata = report_data['metadata']
        personal_info = report_data['personal_info']
        
        # Title
        title = Paragraph("CV Evaluation Report", self.styles['CustomTitle'])
        story.append(title)
        
        # Candidate name
        if personal_info.get('name'):
            candidate_name = Paragraph(f"<b>{personal_info['name']}</b>", self.styles['Heading2'])
            story.append(candidate_name)
        
        # Generation info
        gen_date = metadata['generated_at'].strftime('%Y-%m-%d %H:%M:%S')
        gen_info = Paragraph(f"Generated on: {gen_date}", self.styles['Normal'])
        story.append(gen_info)
        
        story.append(Spacer(1, 20))
    
    def _add_summary(self, story: List, report_data: Dict[str, Any]):
        """Add executive summary."""
        summary = report_data['summary']
        
        story.append(Paragraph("Executive Summary", self.styles['SectionHeader']))
        
        # Overall score
        score_text = f"<b>Overall Score: {summary['overall_score']:.1f}/100</b>"
        story.append(Paragraph(score_text, self.styles['ScoreText']))
        
        # Fit percentage
        fit_text = f"<b>Job Fit: {summary['fit_percentage']:.1f}%</b>"
        story.append(Paragraph(fit_text, self.styles['ScoreText']))
        
        # Summary table
        summary_data = [
            ['Metric', 'Value'],
            ['Strengths Identified', str(summary['strengths_count'])],
            ['Areas for Improvement', str(summary['weaknesses_count'])],
            ['Recommendations', str(summary['recommendations_count'])]
        ]
        
        summary_table = Table(summary_data, colWidths=[3*inch, 2*inch])
        summary_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2c5aa0')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        
        story.append(summary_table)
        story.append(Spacer(1, 20))
    
    def _add_section_scores(self, story: List, report_data: Dict[str, Any]):
        """Add section scores breakdown."""
        story.append(Paragraph("Section Scores", self.styles['SectionHeader']))
        
        # Create scores table
        scores_data = [['Section', 'Score', 'Percentage', 'Feedback']]
        
        for score in report_data['section_scores']:
            scores_data.append([
                score['section'],
                f"{score['score']:.1f}/{score['max_score']:.1f}",
                f"{score['percentage']:.1f}%",
                score['feedback'][:50] + "..." if len(score['feedback']) > 50 else score['feedback']
            ])
        
        scores_table = Table(scores_data, colWidths=[1.5*inch, 1*inch, 1*inch, 3*inch])
        scores_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2c5aa0')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 10),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('FONTSIZE', (0, 1), (-1, -1), 9),
            ('ALIGN', (3, 1), (3, -1), 'LEFT')  # Left align feedback column
        ]))
        
        story.append(scores_table)
        story.append(Spacer(1, 20))
    
    def _add_detailed_analysis(self, story: List, report_data: Dict[str, Any]):
        """Add detailed analysis section."""
        analysis = report_data['analysis']
        
        story.append(Paragraph("Detailed Analysis", self.styles['SectionHeader']))
        
        # Strengths
        if analysis['strengths']:
            story.append(Paragraph("<b>Strengths:</b>", self.styles['Heading3']))
            for strength in analysis['strengths']:
                story.append(Paragraph(f"• {strength}", self.styles['Normal']))
            story.append(Spacer(1, 10))
        
        # Weaknesses
        if analysis['weaknesses']:
            story.append(Paragraph("<b>Areas for Improvement:</b>", self.styles['Heading3']))
            for weakness in analysis['weaknesses']:
                story.append(Paragraph(f"• {weakness}", self.styles['Normal']))
            story.append(Spacer(1, 10))
    
    def _add_skills_analysis(self, story: List, report_data: Dict[str, Any]):
        """Add skills analysis section."""
        skills = report_data['skills']
        
        if not skills:
            return
        
        story.append(Paragraph("Skills Analysis", self.styles['SectionHeader']))
        
        # Group skills by category
        skills_by_category = {}
        for skill in skills:
            category = skill['category']
            if category not in skills_by_category:
                skills_by_category[category] = []
            skills_by_category[category].append(skill)
        
        # Create skills table
        skills_data = [['Skill', 'Category', 'Level', 'Experience', 'Confidence']]
        
        for skill in skills[:15]:  # Limit to top 15 skills
            skills_data.append([
                skill['name'],
                skill['category'],
                skill['level'],
                f"{skill['years_experience']} years" if skill['years_experience'] else "N/A",
                f"{skill['confidence']:.1f}"
            ])
        
        skills_table = Table(skills_data, colWidths=[1.5*inch, 1*inch, 1*inch, 1*inch, 1*inch])
        skills_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2c5aa0')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 9),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('FONTSIZE', (0, 1), (-1, -1), 8)
        ]))
        
        story.append(skills_table)
        story.append(Spacer(1, 20))
    
    def _add_experience_analysis(self, story: List, report_data: Dict[str, Any]):
        """Add work experience analysis."""
        experience = report_data['experience']
        
        if not experience:
            return
        
        story.append(Paragraph("Work Experience Analysis", self.styles['SectionHeader']))
        
        for exp in experience:
            # Company and position
            exp_header = f"<b>{exp['position']}</b> at <b>{exp['company']}</b>"
            story.append(Paragraph(exp_header, self.styles['Heading4']))
            
            # Duration
            if exp['start_date'] and exp['end_date']:
                duration = f"{exp['start_date']} - {exp['end_date']}"
                if exp['duration_months']:
                    duration += f" ({exp['duration_months']} months)"
                story.append(Paragraph(duration, self.styles['Normal']))
            
            # Technologies
            if exp['technologies']:
                tech_text = f"<b>Technologies:</b> {', '.join(exp['technologies'][:10])}"
                story.append(Paragraph(tech_text, self.styles['Normal']))
            
            # Key achievements
            if exp['achievements']:
                story.append(Paragraph("<b>Key Achievements:</b>", self.styles['Normal']))
                for achievement in exp['achievements'][:3]:
                    story.append(Paragraph(f"• {achievement}", self.styles['Normal']))
            
            story.append(Spacer(1, 10))
    
    def _add_education_analysis(self, story: List, report_data: Dict[str, Any]):
        """Add education analysis."""
        education = report_data['education']
        
        if not education:
            return
        
        story.append(Paragraph("Education Analysis", self.styles['SectionHeader']))
        
        # Education table
        edu_data = [['Institution', 'Degree', 'Field of Study', 'Year']]
        
        for edu in education:
            edu_data.append([
                edu['institution'],
                edu['degree'],
                edu['field_of_study'] or 'N/A',
                str(edu['graduation_year']) if edu['graduation_year'] else 'N/A'
            ])
        
        edu_table = Table(edu_data, colWidths=[2*inch, 1.5*inch, 1.5*inch, 1*inch])
        edu_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2c5aa0')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 9),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('FONTSIZE', (0, 1), (-1, -1), 8)
        ]))
        
        story.append(edu_table)
        story.append(Spacer(1, 20))
    
    def _add_recommendations(self, story: List, report_data: Dict[str, Any]):
        """Add recommendations section."""
        recommendations = report_data['analysis']['recommendations']
        
        if not recommendations:
            return
        
        story.append(Paragraph("Recommendations", self.styles['SectionHeader']))
        
        for i, recommendation in enumerate(recommendations, 1):
            rec_text = f"<b>{i}.</b> {recommendation}"
            story.append(Paragraph(rec_text, self.styles['Normal']))
            story.append(Spacer(1, 5))
        
        story.append(Spacer(1, 20))
