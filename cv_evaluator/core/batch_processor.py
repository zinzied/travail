"""
Batch processing capabilities for multiple CV evaluations.
"""

import logging
import asyncio
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Dict, Any, Optional, Callable
from datetime import datetime
import json
from tqdm import tqdm
from ..core.models import CVAnalysisResult, EvaluationCriteria
from ..core.evaluator import CVEvaluator
from ..utils.exceptions import CVEvaluatorError
from ..utils.config import config

logger = logging.getLogger(__name__)


class BatchProcessor:
    """Handle batch processing of multiple CVs with progress tracking."""
    
    def __init__(self, 
                 evaluation_criteria: Optional[EvaluationCriteria] = None,
                 max_workers: Optional[int] = None,
                 progress_callback: Optional[Callable] = None):
        """
        Initialize batch processor.
        
        Args:
            evaluation_criteria: Evaluation criteria to use
            max_workers: Maximum number of concurrent workers
            progress_callback: Optional callback for progress updates
        """
        self.evaluation_criteria = evaluation_criteria
        self.max_workers = max_workers or config.get('batch.max_concurrent_jobs', 5)
        self.progress_callback = progress_callback
        self.evaluator = CVEvaluator(evaluation_criteria)
        
    def process_directory(self, 
                         input_dir: str,
                         output_dir: str,
                         file_pattern: str = "*.pdf",
                         generate_reports: bool = True,
                         report_format: str = "pdf") -> Dict[str, Any]:
        """
        Process all CV files in a directory.
        
        Args:
            input_dir: Directory containing CV files
            output_dir: Directory for output files
            file_pattern: File pattern to match (e.g., "*.pdf")
            generate_reports: Whether to generate individual reports
            report_format: Format for reports ("pdf", "word", "html")
            
        Returns:
            Dictionary with batch processing results
        """
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        
        if not input_path.exists():
            raise CVEvaluatorError(f"Input directory does not exist: {input_dir}")
        
        # Create output directory
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Find CV files
        cv_files = list(input_path.glob(file_pattern))
        if not cv_files:
            raise CVEvaluatorError(f"No files found matching pattern: {file_pattern}")
        
        logger.info(f"Found {len(cv_files)} CV files to process")
        
        # Process files
        results = self._process_files_parallel(
            cv_files, output_path, generate_reports, report_format
        )
        
        # Generate summary report
        summary_path = self._generate_summary_report(results, output_path)
        
        return {
            'total_files': len(cv_files),
            'successful': len([r for r in results if r['success']]),
            'failed': len([r for r in results if not r['success']]),
            'results': results,
            'summary_report': summary_path,
            'output_directory': str(output_path)
        }
    
    def process_file_list(self,
                         file_paths: List[str],
                         output_dir: str,
                         generate_reports: bool = True,
                         report_format: str = "pdf") -> Dict[str, Any]:
        """
        Process a specific list of CV files.
        
        Args:
            file_paths: List of CV file paths
            output_dir: Directory for output files
            generate_reports: Whether to generate individual reports
            report_format: Format for reports
            
        Returns:
            Dictionary with batch processing results
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Convert to Path objects and validate
        cv_files = []
        for file_path in file_paths:
            path = Path(file_path)
            if path.exists():
                cv_files.append(path)
            else:
                logger.warning(f"File not found: {file_path}")
        
        if not cv_files:
            raise CVEvaluatorError("No valid CV files found")
        
        logger.info(f"Processing {len(cv_files)} CV files")
        
        # Process files
        results = self._process_files_parallel(
            cv_files, output_path, generate_reports, report_format
        )
        
        # Generate summary report
        summary_path = self._generate_summary_report(results, output_path)
        
        return {
            'total_files': len(cv_files),
            'successful': len([r for r in results if r['success']]),
            'failed': len([r for r in results if not r['success']]),
            'results': results,
            'summary_report': summary_path,
            'output_directory': str(output_path)
        }
    
    def _process_files_parallel(self,
                               cv_files: List[Path],
                               output_dir: Path,
                               generate_reports: bool,
                               report_format: str) -> List[Dict[str, Any]]:
        """Process CV files in parallel using ThreadPoolExecutor."""
        results = []
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks
            future_to_file = {
                executor.submit(
                    self._process_single_file,
                    cv_file,
                    output_dir,
                    generate_reports,
                    report_format
                ): cv_file
                for cv_file in cv_files
            }
            
            # Process completed tasks with progress bar
            with tqdm(total=len(cv_files), desc="Processing CVs") as pbar:
                for future in as_completed(future_to_file):
                    cv_file = future_to_file[future]
                    try:
                        result = future.result()
                        results.append(result)
                        
                        # Update progress
                        pbar.update(1)
                        if self.progress_callback:
                            self.progress_callback(len(results), len(cv_files), result)
                            
                    except Exception as e:
                        logger.error(f"Failed to process {cv_file}: {e}")
                        results.append({
                            'file_path': str(cv_file),
                            'success': False,
                            'error': str(e),
                            'processing_time': 0
                        })
                        pbar.update(1)
        
        return results
    
    def _process_single_file(self,
                           cv_file: Path,
                           output_dir: Path,
                           generate_reports: bool,
                           report_format: str) -> Dict[str, Any]:
        """Process a single CV file."""
        start_time = datetime.now()
        
        try:
            logger.info(f"Processing: {cv_file.name}")
            
            # Evaluate CV
            analysis_result = self.evaluator.evaluate_cv(str(cv_file))
            
            # Generate individual report if requested
            report_path = None
            if generate_reports:
                report_filename = f"{cv_file.stem}_evaluation.{report_format}"
                report_path = output_dir / report_filename
                
                self.evaluator.generate_report(
                    analysis_result,
                    str(report_path),
                    format=report_format
                )
            
            # Save analysis result as JSON
            json_path = output_dir / f"{cv_file.stem}_analysis.json"
            self._save_analysis_json(analysis_result, json_path)
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            return {
                'file_path': str(cv_file),
                'success': True,
                'overall_score': analysis_result.overall_score,
                'fit_percentage': analysis_result.fit_percentage,
                'candidate_name': analysis_result.cv_data.personal_info.name,
                'report_path': str(report_path) if report_path else None,
                'json_path': str(json_path),
                'processing_time': processing_time,
                'analysis_result': analysis_result
            }
            
        except Exception as e:
            processing_time = (datetime.now() - start_time).total_seconds()
            logger.error(f"Failed to process {cv_file}: {e}")
            
            return {
                'file_path': str(cv_file),
                'success': False,
                'error': str(e),
                'processing_time': processing_time
            }
    
    def _save_analysis_json(self, analysis_result: CVAnalysisResult, output_path: Path):
        """Save analysis result as JSON file."""
        try:
            # Convert to dictionary for JSON serialization
            result_dict = {
                'metadata': {
                    'analysis_timestamp': analysis_result.analysis_timestamp.isoformat(),
                    'overall_score': analysis_result.overall_score,
                    'fit_percentage': analysis_result.fit_percentage
                },
                'personal_info': {
                    'name': analysis_result.cv_data.personal_info.name,
                    'email': analysis_result.cv_data.personal_info.email,
                    'phone': analysis_result.cv_data.personal_info.phone
                },
                'section_scores': [
                    {
                        'section': score.section,
                        'score': score.score,
                        'max_score': score.max_score,
                        'percentage': (score.score / score.max_score) * 100,
                        'feedback': score.feedback
                    }
                    for score in analysis_result.section_scores
                ],
                'skills': [
                    {
                        'name': skill.name,
                        'category': skill.category,
                        'level': skill.level.value if skill.level else None,
                        'confidence': skill.confidence
                    }
                    for skill in analysis_result.cv_data.skills
                ],
                'analysis': {
                    'strengths': analysis_result.strengths,
                    'weaknesses': analysis_result.weaknesses,
                    'recommendations': analysis_result.recommendations
                }
            }
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(result_dict, f, indent=2, ensure_ascii=False)
                
        except Exception as e:
            logger.error(f"Failed to save JSON analysis: {e}")
    
    def _generate_summary_report(self, results: List[Dict[str, Any]], 
                               output_dir: Path) -> str:
        """Generate a summary report for the batch processing."""
        try:
            summary_path = output_dir / "batch_summary.json"
            
            # Calculate statistics
            successful_results = [r for r in results if r['success']]
            failed_results = [r for r in results if not r['success']]
            
            if successful_results:
                scores = [r['overall_score'] for r in successful_results]
                fit_percentages = [r['fit_percentage'] for r in successful_results]
                
                statistics = {
                    'total_processed': len(results),
                    'successful': len(successful_results),
                    'failed': len(failed_results),
                    'success_rate': len(successful_results) / len(results) * 100,
                    'score_statistics': {
                        'average': sum(scores) / len(scores),
                        'minimum': min(scores),
                        'maximum': max(scores),
                        'median': sorted(scores)[len(scores) // 2]
                    },
                    'fit_statistics': {
                        'average': sum(fit_percentages) / len(fit_percentages),
                        'minimum': min(fit_percentages),
                        'maximum': max(fit_percentages)
                    },
                    'top_candidates': sorted(
                        successful_results,
                        key=lambda x: x['overall_score'],
                        reverse=True
                    )[:5],
                    'processing_time': sum(r['processing_time'] for r in results),
                    'generated_at': datetime.now().isoformat()
                }
            else:
                statistics = {
                    'total_processed': len(results),
                    'successful': 0,
                    'failed': len(failed_results),
                    'success_rate': 0,
                    'errors': [r['error'] for r in failed_results],
                    'generated_at': datetime.now().isoformat()
                }
            
            # Add failed files information
            if failed_results:
                statistics['failed_files'] = [
                    {
                        'file_path': r['file_path'],
                        'error': r['error']
                    }
                    for r in failed_results
                ]
            
            with open(summary_path, 'w', encoding='utf-8') as f:
                json.dump(statistics, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Batch summary generated: {summary_path}")
            return str(summary_path)
            
        except Exception as e:
            logger.error(f"Failed to generate summary report: {e}")
            return ""
