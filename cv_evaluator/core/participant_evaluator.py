"""
Participant evaluation system for handling multiple files per candidate.

Created by: Zied Boughdir (@zinzied)
GitHub: https://github.com/zinzied/cv-evaluation-system
"""

import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
from datetime import datetime
import json

from .models import CVAnalysisResult, EvaluationCriteria, CVData, PersonalInfo
from .evaluator import CVEvaluator
from ..pdf.extractor import PDFExtractor
from ..pdf.parser import CVParser
from ..utils.exceptions import CVEvaluatorError

logger = logging.getLogger(__name__)


class ParticipantFile:
    """Represents a single file for a participant."""
    
    def __init__(self, file_path: str, file_type: str, description: str = ""):
        self.file_path = Path(file_path)
        self.file_type = file_type  # 'cv', 'cover_letter', 'portfolio', 'transcript', 'other'
        self.description = description
        self.content = ""
        self.extracted_data = None
        self.processing_status = "pending"
        self.error_message = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'file_path': str(self.file_path),
            'file_type': self.file_type,
            'description': self.description,
            'processing_status': self.processing_status,
            'error_message': self.error_message
        }


class Participant:
    """Represents a participant with multiple files."""
    
    def __init__(self, participant_id: str, name: str = ""):
        self.participant_id = participant_id
        self.name = name
        self.files: List[ParticipantFile] = []
        self.combined_cv_data: Optional[CVData] = None
        self.evaluation_result: Optional[CVAnalysisResult] = None
        self.evaluation_timestamp: Optional[datetime] = None
        self.notes = ""
    
    def add_file(self, file_path: str, file_type: str, description: str = ""):
        """Add a file to the participant."""
        participant_file = ParticipantFile(file_path, file_type, description)
        self.files.append(participant_file)
        logger.info(f"Added file {file_path} ({file_type}) for participant {self.participant_id}")
    
    def get_files_by_type(self, file_type: str) -> List[ParticipantFile]:
        """Get all files of a specific type."""
        return [f for f in self.files if f.file_type == file_type]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'participant_id': self.participant_id,
            'name': self.name,
            'files': [f.to_dict() for f in self.files],
            'evaluation_timestamp': self.evaluation_timestamp.isoformat() if self.evaluation_timestamp else None,
            'notes': self.notes
        }


class ParticipantEvaluator:
    """Evaluate participants with multiple files using custom criteria."""
    
    def __init__(self, evaluation_criteria: Optional[EvaluationCriteria] = None):
        self.evaluation_criteria = evaluation_criteria
        self.pdf_extractor = PDFExtractor()
        self.cv_parser = CVParser()
        self.evaluator = CVEvaluator(evaluation_criteria)
        self.participants: Dict[str, Participant] = {}
    
    def add_participant(self, participant_id: str, name: str = "") -> Participant:
        """Add a new participant."""
        participant = Participant(participant_id, name)
        self.participants[participant_id] = participant
        logger.info(f"Added participant: {participant_id} ({name})")
        return participant
    
    def add_participant_files(self, participant_id: str, files: List[Dict[str, str]]):
        """
        Add multiple files for a participant.
        
        Args:
            participant_id: Unique identifier for the participant
            files: List of dictionaries with 'path', 'type', and optional 'description'
        """
        if participant_id not in self.participants:
            self.add_participant(participant_id)
        
        participant = self.participants[participant_id]
        
        for file_info in files:
            file_path = file_info['path']
            file_type = file_info.get('type', 'other')
            description = file_info.get('description', '')
            
            if Path(file_path).exists():
                participant.add_file(file_path, file_type, description)
            else:
                logger.warning(f"File not found: {file_path}")
    
    def process_participant_files(self, participant_id: str) -> bool:
        """Process all files for a participant."""
        if participant_id not in self.participants:
            logger.error(f"Participant not found: {participant_id}")
            return False
        
        participant = self.participants[participant_id]
        logger.info(f"Processing files for participant: {participant_id}")
        
        # Process each file
        for file_obj in participant.files:
            try:
                self._process_single_file(file_obj)
            except Exception as e:
                file_obj.processing_status = "error"
                file_obj.error_message = str(e)
                logger.error(f"Failed to process file {file_obj.file_path}: {e}")
        
        # Combine data from all files
        self._combine_participant_data(participant)
        
        return True
    
    def _process_single_file(self, file_obj: ParticipantFile):
        """Process a single file."""
        file_obj.processing_status = "processing"
        
        try:
            if file_obj.file_path.suffix.lower() == '.pdf':
                # Extract text from PDF
                extraction_result = self.pdf_extractor.extract_text(str(file_obj.file_path))
                file_obj.content = extraction_result.get('text', '')
                
                if file_obj.file_type == 'cv':
                    # Parse CV structure
                    file_obj.extracted_data = self.cv_parser.parse_cv(file_obj.content)
            
            elif file_obj.file_path.suffix.lower() == '.txt':
                # Read text file
                file_obj.content = file_obj.file_path.read_text(encoding='utf-8')
                
                if file_obj.file_type == 'cv':
                    # Parse CV structure
                    file_obj.extracted_data = self.cv_parser.parse_cv(file_obj.content)
            
            elif file_obj.file_path.suffix.lower() == '.json':
                # Read JSON file
                with open(file_obj.file_path, 'r') as f:
                    json_data = json.load(f)
                file_obj.content = json.dumps(json_data, indent=2)
                file_obj.extracted_data = json_data
            
            else:
                logger.warning(f"Unsupported file type: {file_obj.file_path.suffix}")
                file_obj.content = f"Unsupported file type: {file_obj.file_path.suffix}"
            
            file_obj.processing_status = "completed"
            
        except Exception as e:
            file_obj.processing_status = "error"
            file_obj.error_message = str(e)
            raise
    
    def _combine_participant_data(self, participant: Participant):
        """Combine data from all participant files into a single CV data object."""
        # Start with CV files
        cv_files = participant.get_files_by_type('cv')
        
        if cv_files:
            # Use the first CV as base
            main_cv = cv_files[0]
            if main_cv.extracted_data:
                combined_data = main_cv.extracted_data
            else:
                # Create basic CV data from text
                combined_data = self.cv_parser.parse_cv(main_cv.content)
        else:
            # Create empty CV data
            combined_data = CVData(personal_info=PersonalInfo())
        
        # Update participant name if found in CV
        if combined_data.personal_info.name and not participant.name:
            participant.name = combined_data.personal_info.name
        
        # Enhance with data from other files
        self._enhance_with_additional_files(combined_data, participant)
        
        participant.combined_cv_data = combined_data
        logger.info(f"Combined data for participant {participant.participant_id}")
    
    def _enhance_with_additional_files(self, cv_data: CVData, participant: Participant):
        """Enhance CV data with information from additional files."""
        # Process cover letters
        cover_letters = participant.get_files_by_type('cover_letter')
        for cover_letter in cover_letters:
            if cover_letter.content:
                # Extract additional skills or information from cover letter
                additional_skills = self.cv_parser._extract_skills(cover_letter.content)
                cv_data.skills.extend(additional_skills)
        
        # Process portfolio files
        portfolios = participant.get_files_by_type('portfolio')
        for portfolio in portfolios:
            if portfolio.content:
                # Add portfolio projects
                projects = self._extract_projects_from_text(portfolio.content)
                cv_data.projects.extend(projects)
        
        # Process transcripts
        transcripts = participant.get_files_by_type('transcript')
        for transcript in transcripts:
            if transcript.content:
                # Extract additional education information
                education = self.cv_parser._extract_education(transcript.content)
                cv_data.education.extend(education)
        
        # Remove duplicates
        cv_data.skills = self._deduplicate_skills(cv_data.skills)
        cv_data.projects = list(set(cv_data.projects))
    
    def _extract_projects_from_text(self, text: str) -> List[str]:
        """Extract project names from text."""
        import re
        
        # Look for project patterns
        project_patterns = [
            r'project:?\s*([^\n]+)',
            r'built\s+([^\n]+)',
            r'developed\s+([^\n]+)',
            r'created\s+([^\n]+)'
        ]
        
        projects = []
        for pattern in project_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            projects.extend([match.strip() for match in matches if len(match.strip()) > 5])
        
        return projects[:5]  # Limit to 5 projects
    
    def _deduplicate_skills(self, skills):
        """Remove duplicate skills."""
        seen = set()
        unique_skills = []
        
        for skill in skills:
            skill_name_lower = skill.name.lower()
            if skill_name_lower not in seen:
                seen.add(skill_name_lower)
                unique_skills.append(skill)
        
        return unique_skills
    
    def evaluate_participant(self, participant_id: str) -> Optional[CVAnalysisResult]:
        """Evaluate a participant using the combined data."""
        if participant_id not in self.participants:
            logger.error(f"Participant not found: {participant_id}")
            return None
        
        participant = self.participants[participant_id]
        
        if not participant.combined_cv_data:
            logger.error(f"No processed data for participant: {participant_id}")
            return None
        
        try:
            # Evaluate using the combined CV data
            result = self.evaluator.analyzer.analyze_cv(participant.combined_cv_data)
            
            participant.evaluation_result = result
            participant.evaluation_timestamp = datetime.now()
            
            logger.info(f"Evaluated participant {participant_id}: {result.overall_score:.1f}/100")
            return result
            
        except Exception as e:
            logger.error(f"Failed to evaluate participant {participant_id}: {e}")
            return None
    
    def evaluate_all_participants(self) -> Dict[str, CVAnalysisResult]:
        """Evaluate all participants."""
        results = {}
        
        for participant_id in self.participants:
            # Process files first
            self.process_participant_files(participant_id)
            
            # Then evaluate
            result = self.evaluate_participant(participant_id)
            if result:
                results[participant_id] = result
        
        return results
    
    def get_participant_summary(self, participant_id: str) -> Dict[str, Any]:
        """Get a summary of participant evaluation."""
        if participant_id not in self.participants:
            return {}
        
        participant = self.participants[participant_id]
        
        summary = {
            'participant_id': participant_id,
            'name': participant.name,
            'total_files': len(participant.files),
            'files_by_type': {},
            'processing_status': {},
            'evaluation_completed': participant.evaluation_result is not None
        }
        
        # Count files by type
        for file_obj in participant.files:
            file_type = file_obj.file_type
            summary['files_by_type'][file_type] = summary['files_by_type'].get(file_type, 0) + 1
            summary['processing_status'][file_obj.file_type] = file_obj.processing_status
        
        # Add evaluation results if available
        if participant.evaluation_result:
            summary.update({
                'overall_score': participant.evaluation_result.overall_score,
                'fit_percentage': participant.evaluation_result.fit_percentage,
                'evaluation_timestamp': participant.evaluation_timestamp.isoformat()
            })
        
        return summary
    
    def export_results(self, output_path: str):
        """Export all participant results to JSON."""
        results = {
            'evaluation_criteria': {
                'required_skills': self.evaluation_criteria.required_skills if self.evaluation_criteria else [],
                'preferred_skills': self.evaluation_criteria.preferred_skills if self.evaluation_criteria else [],
                'min_experience_years': self.evaluation_criteria.min_experience_years if self.evaluation_criteria else 0
            },
            'participants': {}
        }
        
        for participant_id, participant in self.participants.items():
            participant_data = participant.to_dict()
            
            if participant.evaluation_result:
                participant_data['evaluation'] = {
                    'overall_score': participant.evaluation_result.overall_score,
                    'fit_percentage': participant.evaluation_result.fit_percentage,
                    'section_scores': [
                        {
                            'section': score.section,
                            'score': score.score,
                            'max_score': score.max_score,
                            'feedback': score.feedback
                        }
                        for score in participant.evaluation_result.section_scores
                    ],
                    'strengths': participant.evaluation_result.strengths,
                    'weaknesses': participant.evaluation_result.weaknesses,
                    'recommendations': participant.evaluation_result.recommendations
                }
            
            results['participants'][participant_id] = participant_data
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Results exported to: {output_path}")
        return output_path
