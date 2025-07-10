"""
Chart generation utilities for reports.
"""

import logging
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime
import tempfile
from ..core.models import SectionScore, Skill, WorkExperience

logger = logging.getLogger(__name__)


class ChartGenerator:
    """Generate charts for CV evaluation reports."""
    
    def __init__(self, output_dir: Optional[str] = None):
        self.output_dir = Path(output_dir) if output_dir else Path(tempfile.gettempdir())
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
    
    def create_score_breakdown_chart(self, section_scores: List[SectionScore]) -> str:
        """
        Create a bar chart showing score breakdown by section.
        
        Args:
            section_scores: List of section scores
            
        Returns:
            Path to generated chart image
        """
        try:
            fig, ax = plt.subplots(figsize=(10, 6))
            
            sections = [score.section.title() for score in section_scores]
            scores = [score.score for score in section_scores]
            max_scores = [score.max_score for score in section_scores]
            
            x = np.arange(len(sections))
            width = 0.35
            
            # Create bars
            bars1 = ax.bar(x - width/2, scores, width, label='Actual Score', color='#2c5aa0')
            bars2 = ax.bar(x + width/2, max_scores, width, label='Max Score', color='lightgray', alpha=0.7)
            
            # Customize chart
            ax.set_xlabel('Sections')
            ax.set_ylabel('Score')
            ax.set_title('CV Evaluation Score Breakdown')
            ax.set_xticks(x)
            ax.set_xticklabels(sections)
            ax.legend()
            
            # Add value labels on bars
            for bar in bars1:
                height = bar.get_height()
                ax.annotate(f'{height:.1f}',
                           xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, 3),  # 3 points vertical offset
                           textcoords="offset points",
                           ha='center', va='bottom')
            
            plt.tight_layout()
            
            # Save chart
            chart_path = self.output_dir / f"score_breakdown_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            plt.savefig(chart_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Score breakdown chart generated: {chart_path}")
            return str(chart_path)
            
        except Exception as e:
            logger.error(f"Failed to generate score breakdown chart: {e}")
            return ""
    
    def create_skills_radar_chart(self, skills: List[Skill]) -> str:
        """
        Create a radar chart showing skills by category.
        
        Args:
            skills: List of skills
            
        Returns:
            Path to generated chart image
        """
        try:
            # Group skills by category and calculate average confidence
            skill_categories = {}
            for skill in skills:
                category = skill.category or 'General'
                if category not in skill_categories:
                    skill_categories[category] = []
                skill_categories[category].append(skill.confidence)
            
            # Calculate average confidence per category
            category_scores = {}
            for category, confidences in skill_categories.items():
                category_scores[category] = np.mean(confidences) * 100  # Convert to percentage
            
            if len(category_scores) < 3:
                # Not enough categories for radar chart
                return self._create_skills_bar_chart(skills)
            
            # Prepare data for radar chart
            categories = list(category_scores.keys())
            values = list(category_scores.values())
            
            # Number of variables
            N = len(categories)
            
            # Compute angle for each axis
            angles = [n / float(N) * 2 * np.pi for n in range(N)]
            angles += angles[:1]  # Complete the circle
            
            # Add first value at the end to close the circle
            values += values[:1]
            
            # Create radar chart
            fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection='polar'))
            
            # Plot
            ax.plot(angles, values, 'o-', linewidth=2, label='Skill Confidence', color='#2c5aa0')
            ax.fill(angles, values, alpha=0.25, color='#2c5aa0')
            
            # Add category labels
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(categories)
            
            # Set y-axis limits
            ax.set_ylim(0, 100)
            ax.set_ylabel('Confidence (%)', labelpad=30)
            
            # Add title
            ax.set_title('Skills Radar Chart', size=16, fontweight='bold', pad=20)
            
            # Add grid
            ax.grid(True)
            
            plt.tight_layout()
            
            # Save chart
            chart_path = self.output_dir / f"skills_radar_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            plt.savefig(chart_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Skills radar chart generated: {chart_path}")
            return str(chart_path)
            
        except Exception as e:
            logger.error(f"Failed to generate skills radar chart: {e}")
            return ""
    
    def _create_skills_bar_chart(self, skills: List[Skill]) -> str:
        """Create a bar chart for skills when radar chart is not suitable."""
        try:
            # Group skills by category
            skill_categories = {}
            for skill in skills:
                category = skill.category or 'General'
                if category not in skill_categories:
                    skill_categories[category] = []
                skill_categories[category].append(skill.confidence)
            
            # Calculate average confidence per category
            categories = list(skill_categories.keys())
            avg_confidences = [np.mean(confidences) * 100 for confidences in skill_categories.values()]
            
            # Create bar chart
            fig, ax = plt.subplots(figsize=(10, 6))
            
            bars = ax.bar(categories, avg_confidences, color='#2c5aa0', alpha=0.7)
            
            # Customize chart
            ax.set_xlabel('Skill Categories')
            ax.set_ylabel('Average Confidence (%)')
            ax.set_title('Skills by Category')
            ax.set_ylim(0, 100)
            
            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                ax.annotate(f'{height:.1f}%',
                           xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, 3),
                           textcoords="offset points",
                           ha='center', va='bottom')
            
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            
            # Save chart
            chart_path = self.output_dir / f"skills_bar_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            plt.savefig(chart_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            return str(chart_path)
            
        except Exception as e:
            logger.error(f"Failed to generate skills bar chart: {e}")
            return ""
    
    def create_experience_timeline(self, work_experience: List[WorkExperience]) -> str:
        """
        Create a timeline chart showing work experience.
        
        Args:
            work_experience: List of work experiences
            
        Returns:
            Path to generated chart image
        """
        try:
            if not work_experience:
                return ""
            
            # Prepare data
            companies = []
            positions = []
            start_years = []
            durations = []
            
            for exp in work_experience:
                if exp.start_date and exp.duration_months:
                    try:
                        # Extract year from start_date
                        start_year = int(exp.start_date.split('-')[0]) if '-' in exp.start_date else int(exp.start_date[:4])
                        start_years.append(start_year)
                        companies.append(exp.company[:20] + "..." if len(exp.company) > 20 else exp.company)
                        positions.append(exp.position[:25] + "..." if len(exp.position) > 25 else exp.position)
                        durations.append(exp.duration_months / 12)  # Convert to years
                    except (ValueError, IndexError):
                        continue
            
            if not start_years:
                return ""
            
            # Create timeline chart
            fig, ax = plt.subplots(figsize=(12, max(6, len(companies) * 0.8)))
            
            # Create horizontal bars
            y_positions = range(len(companies))
            bars = ax.barh(y_positions, durations, left=start_years, height=0.6, 
                          color='#2c5aa0', alpha=0.7)
            
            # Customize chart
            ax.set_yticks(y_positions)
            ax.set_yticklabels([f"{pos}\n{comp}" for pos, comp in zip(positions, companies)])
            ax.set_xlabel('Year')
            ax.set_title('Work Experience Timeline')
            
            # Add duration labels
            for i, (bar, duration) in enumerate(zip(bars, durations)):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_y() + bar.get_height()/2,
                       f'{duration:.1f}y', ha='center', va='center', fontweight='bold')
            
            # Set x-axis limits
            if start_years:
                min_year = min(start_years) - 1
                max_year = max([start + dur for start, dur in zip(start_years, durations)]) + 1
                ax.set_xlim(min_year, max_year)
            
            plt.tight_layout()
            
            # Save chart
            chart_path = self.output_dir / f"experience_timeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            plt.savefig(chart_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Experience timeline chart generated: {chart_path}")
            return str(chart_path)
            
        except Exception as e:
            logger.error(f"Failed to generate experience timeline chart: {e}")
            return ""
    
    def create_comparison_chart(self, analysis_results: List[Any]) -> str:
        """
        Create a comparison chart for multiple candidates.
        
        Args:
            analysis_results: List of CV analysis results
            
        Returns:
            Path to generated chart image
        """
        try:
            if len(analysis_results) < 2:
                return ""
            
            # Prepare data
            candidate_names = []
            overall_scores = []
            
            for i, result in enumerate(analysis_results):
                name = result.cv_data.personal_info.name or f"Candidate {i+1}"
                candidate_names.append(name[:15] + "..." if len(name) > 15 else name)
                overall_scores.append(result.overall_score)
            
            # Create comparison chart
            fig, ax = plt.subplots(figsize=(10, 6))
            
            bars = ax.bar(candidate_names, overall_scores, color='#2c5aa0', alpha=0.7)
            
            # Customize chart
            ax.set_xlabel('Candidates')
            ax.set_ylabel('Overall Score')
            ax.set_title('Candidate Comparison')
            ax.set_ylim(0, 100)
            
            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                ax.annotate(f'{height:.1f}',
                           xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, 3),
                           textcoords="offset points",
                           ha='center', va='bottom')
            
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            
            # Save chart
            chart_path = self.output_dir / f"candidate_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            plt.savefig(chart_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Candidate comparison chart generated: {chart_path}")
            return str(chart_path)
            
        except Exception as e:
            logger.error(f"Failed to generate comparison chart: {e}")
            return ""
