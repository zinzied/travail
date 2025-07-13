"""
Excel integration for CV evaluation system.

Created by: Zied Boughdir (@zinzied)
GitHub: https://github.com/zinzied/cv-evaluation-system
"""

import logging
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
from datetime import datetime
import json

from ..core.models import CVData, CVAnalysisResult, PersonalInfo, Skill, WorkExperience, Education
from ..core.evaluator import CVEvaluator
from ..core.participant_evaluator import ParticipantEvaluator

logger = logging.getLogger(__name__)


class ExcelProcessor:
    """Process Excel files for CV evaluation and export results."""
    
    def __init__(self):
        self.evaluator = None
        self.results = []
    
    def import_candidates_from_excel(self, excel_file: str) -> List[Dict[str, Any]]:
        """
        Import candidate data from Excel file.
        
        Expected columns:
        - candidate_id, name, email, phone, skills, experience, education, cv_file_path
        """
        try:
            df = pd.read_excel(excel_file)
            logger.info(f"Loaded Excel file with {len(df)} candidates")
            
            candidates = []
            for index, row in df.iterrows():
                candidate = {
                    'candidate_id': str(row.get('candidate_id', f'CAND_{index+1:03d}')),
                    'name': str(row.get('name', '')),
                    'email': str(row.get('email', '')),
                    'phone': str(row.get('phone', '')),
                    'skills': str(row.get('skills', '')),
                    'experience': str(row.get('experience', '')),
                    'education': str(row.get('education', '')),
                    'cv_file_path': str(row.get('cv_file_path', '')),
                    'notes': str(row.get('notes', '')),
                    'status': str(row.get('status', 'pending'))
                }
                candidates.append(candidate)
            
            logger.info(f"Imported {len(candidates)} candidates from Excel")
            return candidates
            
        except Exception as e:
            logger.error(f"Failed to import candidates from Excel: {e}")
            raise
    
    def create_cv_data_from_excel_row(self, candidate: Dict[str, Any]) -> CVData:
        """Create CVData object from Excel row data."""
        # Personal info
        personal_info = PersonalInfo(
            name=candidate.get('name', ''),
            email=candidate.get('email', ''),
            phone=candidate.get('phone', '')
        )
        
        # Skills
        skills = []
        skills_text = candidate.get('skills', '')
        if skills_text:
            skill_names = [s.strip() for s in skills_text.split(',') if s.strip()]
            for skill_name in skill_names:
                skills.append(Skill(name=skill_name, category='general', confidence=0.8))
        
        # Work experience
        work_experience = []
        experience_text = candidate.get('experience', '')
        if experience_text:
            # Simple parsing - can be enhanced
            exp_entries = experience_text.split(';')
            for exp in exp_entries:
                if exp.strip():
                    work_experience.append(WorkExperience(
                        position=exp.strip(),
                        company='Unknown',
                        duration_months=12  # Default
                    ))
        
        # Education
        education = []
        education_text = candidate.get('education', '')
        if education_text:
            edu_entries = education_text.split(';')
            for edu in edu_entries:
                if edu.strip():
                    education.append(Education(
                        degree=edu.strip(),
                        institution='Unknown'
                    ))
        
        return CVData(
            personal_info=personal_info,
            skills=skills,
            work_experience=work_experience,
            education=education,
            projects=[],
            languages=[],
            certifications=[]
        )
    
    def evaluate_candidates_from_excel(self, excel_file: str, criteria_name: str = "default") -> List[Dict[str, Any]]:
        """Evaluate all candidates from Excel file."""
        try:
            # Import candidates
            candidates = self.import_candidates_from_excel(excel_file)
            
            # Initialize evaluator
            self.evaluator = CVEvaluator(criteria_name=criteria_name)
            
            results = []
            
            for candidate in candidates:
                try:
                    logger.info(f"Evaluating candidate: {candidate['candidate_id']}")
                    
                    # Check if CV file exists
                    cv_file_path = candidate.get('cv_file_path', '')
                    if cv_file_path and Path(cv_file_path).exists():
                        # Evaluate from CV file
                        result = self.evaluator.evaluate_cv(cv_file_path)
                    else:
                        # Create CV data from Excel row
                        cv_data = self.create_cv_data_from_excel_row(candidate)
                        result = self.evaluator.analyzer.analyze_cv(cv_data)
                    
                    if result:
                        # Combine candidate info with evaluation result
                        candidate_result = {
                            'candidate_id': candidate['candidate_id'],
                            'name': candidate['name'],
                            'email': candidate['email'],
                            'phone': candidate['phone'],
                            'overall_score': result.overall_score,
                            'fit_percentage': result.fit_percentage,
                            'skills_score': next((s.score for s in result.section_scores if s.section == 'skills'), 0),
                            'experience_score': next((s.score for s in result.section_scores if s.section == 'experience'), 0),
                            'education_score': next((s.score for s in result.section_scores if s.section == 'education'), 0),
                            'additional_score': next((s.score for s in result.section_scores if s.section == 'additional'), 0),
                            'strengths': '; '.join(result.strengths[:3]),
                            'weaknesses': '; '.join(result.weaknesses[:3]),
                            'recommendations': '; '.join(result.recommendations[:3]),
                            'skills_found': len(result.cv_data.skills),
                            'experience_years': sum([exp.duration_months or 0 for exp in result.cv_data.work_experience]) / 12,
                            'education_level': len(result.cv_data.education),
                            'evaluation_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                            'status': candidate.get('status', 'evaluated'),
                            'notes': candidate.get('notes', '')
                        }
                        results.append(candidate_result)
                        
                    else:
                        # Failed evaluation
                        candidate_result = {
                            'candidate_id': candidate['candidate_id'],
                            'name': candidate['name'],
                            'email': candidate['email'],
                            'phone': candidate['phone'],
                            'overall_score': 0,
                            'fit_percentage': 0,
                            'skills_score': 0,
                            'experience_score': 0,
                            'education_score': 0,
                            'additional_score': 0,
                            'strengths': 'Evaluation failed',
                            'weaknesses': 'Could not process CV',
                            'recommendations': 'Please check CV file',
                            'skills_found': 0,
                            'experience_years': 0,
                            'education_level': 0,
                            'evaluation_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                            'status': 'failed',
                            'notes': candidate.get('notes', '')
                        }
                        results.append(candidate_result)
                
                except Exception as e:
                    logger.error(f"Failed to evaluate candidate {candidate['candidate_id']}: {e}")
                    continue
            
            self.results = results
            logger.info(f"Completed evaluation of {len(results)} candidates")
            return results
            
        except Exception as e:
            logger.error(f"Failed to evaluate candidates from Excel: {e}")
            raise
    
    def export_results_to_excel(self, results: List[Dict[str, Any]], output_file: str) -> str:
        """Export evaluation results to Excel file."""
        try:
            # Create DataFrame
            df = pd.DataFrame(results)
            
            # Create Excel writer with multiple sheets
            with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
                # Main results sheet
                df.to_excel(writer, sheet_name='Evaluation Results', index=False)
                
                # Summary statistics sheet
                summary_data = self._create_summary_statistics(results)
                summary_df = pd.DataFrame(summary_data)
                summary_df.to_excel(writer, sheet_name='Summary Statistics', index=False)
                
                # Top candidates sheet
                top_candidates = sorted(results, key=lambda x: x['overall_score'], reverse=True)[:10]
                top_df = pd.DataFrame(top_candidates)
                top_df.to_excel(writer, sheet_name='Top Candidates', index=False)
                
                # Skills analysis sheet
                skills_analysis = self._create_skills_analysis(results)
                skills_df = pd.DataFrame(skills_analysis)
                skills_df.to_excel(writer, sheet_name='Skills Analysis', index=False)
            
            logger.info(f"Results exported to Excel: {output_file}")
            return output_file
            
        except Exception as e:
            logger.error(f"Failed to export results to Excel: {e}")
            raise
    
    def _create_summary_statistics(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Create summary statistics for the evaluation results."""
        if not results:
            return []
        
        scores = [r['overall_score'] for r in results if r['overall_score'] > 0]
        
        summary = [
            {'Metric': 'Total Candidates', 'Value': len(results)},
            {'Metric': 'Successfully Evaluated', 'Value': len(scores)},
            {'Metric': 'Failed Evaluations', 'Value': len(results) - len(scores)},
            {'Metric': 'Average Score', 'Value': f"{sum(scores)/len(scores):.1f}" if scores else "0"},
            {'Metric': 'Highest Score', 'Value': f"{max(scores):.1f}" if scores else "0"},
            {'Metric': 'Lowest Score', 'Value': f"{min(scores):.1f}" if scores else "0"},
            {'Metric': 'Candidates Above 70%', 'Value': len([s for s in scores if s >= 70])},
            {'Metric': 'Candidates Above 50%', 'Value': len([s for s in scores if s >= 50])},
            {'Metric': 'Evaluation Date', 'Value': datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        ]
        
        return summary
    
    def _create_skills_analysis(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Create skills analysis from evaluation results."""
        skills_analysis = []
        
        # Analyze score distributions
        score_ranges = [
            ('90-100', len([r for r in results if 90 <= r['overall_score'] <= 100])),
            ('80-89', len([r for r in results if 80 <= r['overall_score'] < 90])),
            ('70-79', len([r for r in results if 70 <= r['overall_score'] < 80])),
            ('60-69', len([r for r in results if 60 <= r['overall_score'] < 70])),
            ('50-59', len([r for r in results if 50 <= r['overall_score'] < 60])),
            ('Below 50', len([r for r in results if r['overall_score'] < 50]))
        ]
        
        for score_range, count in score_ranges:
            skills_analysis.append({
                'Score Range': score_range,
                'Number of Candidates': count,
                'Percentage': f"{(count/len(results)*100):.1f}%" if results else "0%"
            })
        
        return skills_analysis
    
    def create_excel_template(self, output_file: str) -> str:
        """Create an Excel template for importing candidates."""
        template_data = {
            'candidate_id': ['CAND_001', 'CAND_002', 'CAND_003'],
            'name': ['John Doe', 'Jane Smith', 'Mike Johnson'],
            'email': ['john.doe@email.com', 'jane.smith@email.com', 'mike.johnson@email.com'],
            'phone': ['+1-555-0123', '+1-555-0124', '+1-555-0125'],
            'skills': [
                'Python, JavaScript, SQL, Machine Learning',
                'Java, Spring, React, AWS, Docker',
                'C#, .NET, Azure, DevOps, Agile'
            ],
            'experience': [
                'Senior Developer at TechCorp (3 years); Developer at StartupXYZ (2 years)',
                'Full Stack Developer at BigTech (4 years); Junior Developer at SmallCorp (1 year)',
                'Software Engineer at Enterprise Inc (5 years)'
            ],
            'education': [
                'MS Computer Science, Stanford University; BS Software Engineering, UC Berkeley',
                'BS Computer Science, MIT; Coding Bootcamp, General Assembly',
                'MS Information Systems, Carnegie Mellon; BS Mathematics, University of Chicago'
            ],
            'cv_file_path': [
                'path/to/john_doe_cv.pdf',
                'path/to/jane_smith_cv.pdf',
                'path/to/mike_johnson_cv.pdf'
            ],
            'status': ['pending', 'pending', 'pending'],
            'notes': ['Referred by employee', 'Strong technical background', 'Leadership experience']
        }
        
        df = pd.DataFrame(template_data)
        df.to_excel(output_file, index=False)
        
        logger.info(f"Excel template created: {output_file}")
        return output_file
    
    def update_excel_with_results(self, input_file: str, results: List[Dict[str, Any]], output_file: str) -> str:
        """Update existing Excel file with evaluation results."""
        try:
            # Read original Excel file
            df_original = pd.read_excel(input_file)
            
            # Create results DataFrame
            df_results = pd.DataFrame(results)
            
            # Merge on candidate_id
            df_merged = df_original.merge(
                df_results, 
                on='candidate_id', 
                how='left',
                suffixes=('_original', '_evaluated')
            )
            
            # Save updated file
            df_merged.to_excel(output_file, index=False)
            
            logger.info(f"Excel file updated with results: {output_file}")
            return output_file
            
        except Exception as e:
            logger.error(f"Failed to update Excel file: {e}")
            raise


class ExcelBatchProcessor:
    """Process multiple CVs and manage results in Excel format."""
    
    def __init__(self):
        self.processor = ExcelProcessor()
        self.participant_evaluator = None
    
    def process_cv_folder_to_excel(self, cv_folder: str, output_excel: str, criteria_name: str = "default") -> str:
        """Process all CVs in a folder and export results to Excel."""
        try:
            cv_folder_path = Path(cv_folder)
            if not cv_folder_path.exists():
                raise FileNotFoundError(f"CV folder not found: {cv_folder}")
            
            # Find all CV files
            cv_files = []
            for ext in ['*.pdf', '*.txt']:
                cv_files.extend(cv_folder_path.glob(ext))
            
            if not cv_files:
                raise ValueError(f"No CV files found in {cv_folder}")
            
            logger.info(f"Found {len(cv_files)} CV files to process")
            
            # Initialize evaluator
            evaluator = CVEvaluator(criteria_name=criteria_name)
            
            results = []
            for i, cv_file in enumerate(cv_files, 1):
                try:
                    logger.info(f"Processing CV {i}/{len(cv_files)}: {cv_file.name}")
                    
                    # Evaluate CV
                    result = evaluator.evaluate_cv(str(cv_file))
                    
                    if result:
                        candidate_result = {
                            'candidate_id': f"CV_{i:03d}",
                            'name': result.cv_data.personal_info.name or cv_file.stem,
                            'email': result.cv_data.personal_info.email or '',
                            'phone': result.cv_data.personal_info.phone or '',
                            'cv_file': cv_file.name,
                            'overall_score': result.overall_score,
                            'fit_percentage': result.fit_percentage,
                            'skills_score': next((s.score for s in result.section_scores if s.section == 'skills'), 0),
                            'experience_score': next((s.score for s in result.section_scores if s.section == 'experience'), 0),
                            'education_score': next((s.score for s in result.section_scores if s.section == 'education'), 0),
                            'additional_score': next((s.score for s in result.section_scores if s.section == 'additional'), 0),
                            'strengths': '; '.join(result.strengths[:3]),
                            'weaknesses': '; '.join(result.weaknesses[:3]),
                            'recommendations': '; '.join(result.recommendations[:3]),
                            'skills_found': len(result.cv_data.skills),
                            'experience_years': sum([exp.duration_months or 0 for exp in result.cv_data.work_experience]) / 12,
                            'education_level': len(result.cv_data.education),
                            'evaluation_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                            'status': 'evaluated'
                        }
                        results.append(candidate_result)
                    
                except Exception as e:
                    logger.error(f"Failed to process {cv_file.name}: {e}")
                    continue
            
            # Export to Excel
            output_file = self.processor.export_results_to_excel(results, output_excel)
            logger.info(f"Batch processing completed. Results saved to: {output_file}")
            
            return output_file
            
        except Exception as e:
            logger.error(f"Batch processing failed: {e}")
            raise
