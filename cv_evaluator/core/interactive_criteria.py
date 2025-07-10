"""
Interactive criteria definition and management.

Created by: Zied Boughdir (@zinzied)
GitHub: https://github.com/zinzied/cv-evaluation-system
"""

import json
import yaml
from pathlib import Path
from typing import Dict, List, Optional, Any
from rich.console import Console
from rich.prompt import Prompt, Confirm, IntPrompt, FloatPrompt
from rich.table import Table
from rich.panel import Panel
from .models import EvaluationCriteria, EducationLevel

console = Console()


class InteractiveCriteriaBuilder:
    """Interactive builder for evaluation criteria."""
    
    def __init__(self):
        self.console = Console()
        self.criteria_data = {}
    
    def build_criteria_interactively(self) -> EvaluationCriteria:
        """Build evaluation criteria through interactive prompts."""
        console.print(Panel.fit(
            "[bold blue]CV Evaluation Criteria Builder[/bold blue]\n"
            "Let's define your evaluation criteria step by step.",
            title="Welcome"
        ))
        
        # Basic information
        self._get_basic_info()
        
        # Skills requirements
        self._get_skills_requirements()
        
        # Experience requirements
        self._get_experience_requirements()
        
        # Education requirements
        self._get_education_requirements()
        
        # Scoring weights
        self._get_scoring_weights()
        
        # Additional criteria
        self._get_additional_criteria()
        
        # Review and confirm
        if self._review_criteria():
            return EvaluationCriteria(**self.criteria_data)
        else:
            console.print("[yellow]Criteria creation cancelled.[/yellow]")
            return None
    
    def _get_basic_info(self):
        """Get basic information about the evaluation."""
        console.print("\n[bold]Step 1: Basic Information[/bold]")
        
        job_title = Prompt.ask("What position are you evaluating for?", default="General Position")
        department = Prompt.ask("What department/team?", default="")
        
        self.criteria_data.update({
            'job_title': job_title,
            'department': department,
            'max_score': 100
        })
    
    def _get_skills_requirements(self):
        """Get skills requirements."""
        console.print("\n[bold]Step 2: Skills Requirements[/bold]")
        
        # Required skills
        console.print("[cyan]Required Skills (must-have):[/cyan]")
        required_skills = []
        while True:
            skill = Prompt.ask("Enter a required skill (or press Enter to finish)", default="")
            if not skill:
                break
            required_skills.append(skill.strip())
            console.print(f"  ✓ Added: {skill}")
        
        # Preferred skills
        console.print("\n[cyan]Preferred Skills (nice-to-have):[/cyan]")
        preferred_skills = []
        while True:
            skill = Prompt.ask("Enter a preferred skill (or press Enter to finish)", default="")
            if not skill:
                break
            preferred_skills.append(skill.strip())
            console.print(f"  ✓ Added: {skill}")
        
        self.criteria_data.update({
            'required_skills': required_skills,
            'preferred_skills': preferred_skills
        })
    
    def _get_experience_requirements(self):
        """Get experience requirements."""
        console.print("\n[bold]Step 3: Experience Requirements[/bold]")
        
        min_years = IntPrompt.ask("Minimum years of experience required", default=0)
        
        console.print("[cyan]Industry keywords (for relevance scoring):[/cyan]")
        keywords = []
        while True:
            keyword = Prompt.ask("Enter an industry keyword (or press Enter to finish)", default="")
            if not keyword:
                break
            keywords.append(keyword.strip())
            console.print(f"  ✓ Added: {keyword}")
        
        self.criteria_data.update({
            'min_experience_years': min_years,
            'industry_keywords': keywords
        })
    
    def _get_education_requirements(self):
        """Get education requirements."""
        console.print("\n[bold]Step 4: Education Requirements[/bold]")
        
        education_levels = {
            "1": ("high_school", "High School"),
            "2": ("certification", "Certification/Diploma"),
            "3": ("bachelor", "Bachelor's Degree"),
            "4": ("master", "Master's Degree"),
            "5": ("phd", "PhD/Doctorate"),
            "6": ("none", "No specific requirement")
        }
        
        console.print("[cyan]Education level options:[/cyan]")
        for key, (_, label) in education_levels.items():
            console.print(f"  {key}. {label}")
        
        choice = Prompt.ask("Select minimum education level", choices=list(education_levels.keys()), default="6")
        
        if choice != "6":
            level_key, _ = education_levels[choice]
            try:
                self.criteria_data['required_education_level'] = EducationLevel(level_key)
            except ValueError:
                pass  # Skip if invalid
    
    def _get_scoring_weights(self):
        """Get scoring weights for different sections."""
        console.print("\n[bold]Step 5: Scoring Weights[/bold]")
        console.print("[cyan]Define how much each section contributes to the final score (must sum to 1.0):[/cyan]")
        
        weights = {}
        remaining = 1.0
        
        sections = [
            ("skills", "Technical and soft skills"),
            ("experience", "Work experience and career progression"),
            ("education", "Educational background"),
            ("additional", "Languages, certifications, projects")
        ]
        
        for i, (section, description) in enumerate(sections):
            if i == len(sections) - 1:  # Last section gets remaining weight
                weight = remaining
                console.print(f"[yellow]{section.title()} ({description}): {weight:.2f} (remaining)[/yellow]")
            else:
                max_weight = remaining - (len(sections) - i - 1) * 0.05  # Leave at least 0.05 for others
                weight = FloatPrompt.ask(
                    f"{section.title()} ({description}) weight",
                    default=0.4 if section == "skills" else 0.2,
                    show_default=True
                )
                weight = min(weight, max_weight)
                remaining -= weight
            
            weights[section] = weight
        
        # Normalize weights to sum to 1.0
        total = sum(weights.values())
        weights = {k: v/total for k, v in weights.items()}
        
        self.criteria_data['scoring_weights'] = weights
    
    def _get_additional_criteria(self):
        """Get additional evaluation criteria."""
        console.print("\n[bold]Step 6: Additional Criteria[/bold]")
        
        # Language requirements
        if Confirm.ask("Do you have language requirements?", default=False):
            languages = []
            while True:
                lang = Prompt.ask("Enter required language (or press Enter to finish)", default="")
                if not lang:
                    break
                languages.append(lang.strip())
            self.criteria_data['required_languages'] = languages
        
        # Certification preferences
        if Confirm.ask("Do you prefer specific certifications?", default=False):
            certs = []
            while True:
                cert = Prompt.ask("Enter preferred certification (or press Enter to finish)", default="")
                if not cert:
                    break
                certs.append(cert.strip())
            self.criteria_data['preferred_certifications'] = certs
    
    def _review_criteria(self) -> bool:
        """Review and confirm the criteria."""
        console.print("\n[bold]Step 7: Review Your Criteria[/bold]")
        
        # Create review table
        table = Table(title="Evaluation Criteria Summary")
        table.add_column("Category", style="cyan")
        table.add_column("Details", style="white")
        
        # Basic info
        table.add_row("Position", self.criteria_data.get('job_title', 'N/A'))
        table.add_row("Department", self.criteria_data.get('department', 'N/A'))
        
        # Skills
        required_skills = ', '.join(self.criteria_data.get('required_skills', []))
        preferred_skills = ', '.join(self.criteria_data.get('preferred_skills', []))
        table.add_row("Required Skills", required_skills or "None")
        table.add_row("Preferred Skills", preferred_skills or "None")
        
        # Experience
        table.add_row("Min Experience", f"{self.criteria_data.get('min_experience_years', 0)} years")
        keywords = ', '.join(self.criteria_data.get('industry_keywords', []))
        table.add_row("Industry Keywords", keywords or "None")
        
        # Education
        edu_level = self.criteria_data.get('required_education_level')
        table.add_row("Min Education", edu_level.value if edu_level else "No requirement")
        
        # Weights
        weights = self.criteria_data.get('scoring_weights', {})
        weight_str = ', '.join([f"{k}: {v:.1%}" for k, v in weights.items()])
        table.add_row("Scoring Weights", weight_str)
        
        console.print(table)
        
        return Confirm.ask("\nDo you want to save these criteria?", default=True)
    
    def save_criteria_to_file(self, criteria: EvaluationCriteria, filename: str):
        """Save criteria to a YAML file."""
        criteria_dict = {
            'job_title': getattr(criteria, 'job_title', 'Custom Position'),
            'department': getattr(criteria, 'department', ''),
            'required_skills': criteria.required_skills,
            'preferred_skills': criteria.preferred_skills,
            'min_experience_years': criteria.min_experience_years,
            'industry_keywords': criteria.industry_keywords,
            'scoring_weights': criteria.scoring_weights,
            'max_score': criteria.max_score
        }
        
        if criteria.required_education_level:
            criteria_dict['required_education_level'] = criteria.required_education_level.value
        
        # Save to config directory
        config_dir = Path("config")
        config_dir.mkdir(exist_ok=True)
        
        filepath = config_dir / f"{filename}.yaml"
        with open(filepath, 'w') as f:
            yaml.dump(criteria_dict, f, default_flow_style=False, indent=2)
        
        console.print(f"[green]✓ Criteria saved to: {filepath}[/green]")
        return str(filepath)


class CriteriaFromFiles:
    """Extract evaluation criteria from uploaded files."""
    
    def __init__(self):
        self.console = Console()
    
    def extract_criteria_from_files(self, file_paths: List[str]) -> EvaluationCriteria:
        """Extract evaluation criteria from job description or requirement files."""
        console.print(Panel.fit(
            "[bold blue]Extracting Criteria from Files[/bold blue]\n"
            "Analyzing uploaded files to create evaluation criteria.",
            title="File Analysis"
        ))
        
        all_text = ""
        for file_path in file_paths:
            try:
                file_path = Path(file_path)
                if file_path.suffix.lower() == '.txt':
                    all_text += file_path.read_text(encoding='utf-8') + "\n"
                elif file_path.suffix.lower() == '.json':
                    with open(file_path, 'r') as f:
                        data = json.load(f)
                        all_text += str(data) + "\n"
                console.print(f"✓ Processed: {file_path.name}")
            except Exception as e:
                console.print(f"[red]✗ Failed to process {file_path}: {e}[/red]")
        
        if not all_text.strip():
            console.print("[yellow]No text content found in files. Using interactive builder.[/yellow]")
            return InteractiveCriteriaBuilder().build_criteria_interactively()
        
        # Extract criteria from text
        criteria_data = self._analyze_text_for_criteria(all_text)
        
        # Show extracted criteria and allow editing
        return self._review_and_edit_extracted_criteria(criteria_data)
    
    def _analyze_text_for_criteria(self, text: str) -> Dict[str, Any]:
        """Analyze text to extract evaluation criteria."""
        text_lower = text.lower()
        
        # Extract skills
        required_skills = self._extract_skills(text, required=True)
        preferred_skills = self._extract_skills(text, required=False)
        
        # Extract experience requirements
        min_years = self._extract_experience_years(text)
        
        # Extract industry keywords
        keywords = self._extract_keywords(text)
        
        # Default scoring weights
        scoring_weights = {
            'skills': 0.4,
            'experience': 0.3,
            'education': 0.2,
            'additional': 0.1
        }
        
        return {
            'required_skills': required_skills,
            'preferred_skills': preferred_skills,
            'min_experience_years': min_years,
            'industry_keywords': keywords,
            'scoring_weights': scoring_weights,
            'max_score': 100
        }
    
    def _extract_skills(self, text: str, required: bool = True) -> List[str]:
        """Extract skills from text."""
        import re
        
        # Common skill patterns
        skill_patterns = [
            r'python', r'java', r'javascript', r'sql', r'html', r'css',
            r'react', r'angular', r'node\.js', r'django', r'flask',
            r'machine learning', r'data analysis', r'statistics',
            r'aws', r'azure', r'docker', r'kubernetes',
            r'git', r'agile', r'scrum', r'project management',
            r'communication', r'leadership', r'teamwork'
        ]
        
        found_skills = []
        text_lower = text.lower()
        
        for pattern in skill_patterns:
            if re.search(pattern, text_lower):
                # Clean up the skill name
                skill_name = pattern.replace(r'\.', '.').replace(r'\\', '')
                found_skills.append(skill_name)
        
        # Look for explicit requirements/preferences
        if required:
            # Look for "required", "must have", "essential"
            req_patterns = [
                r'required?:?\s*([^.]+)',
                r'must have:?\s*([^.]+)',
                r'essential:?\s*([^.]+)'
            ]
        else:
            # Look for "preferred", "nice to have", "bonus"
            req_patterns = [
                r'preferred?:?\s*([^.]+)',
                r'nice to have:?\s*([^.]+)',
                r'bonus:?\s*([^.]+)'
            ]
        
        for pattern in req_patterns:
            matches = re.findall(pattern, text_lower)
            for match in matches:
                # Extract individual skills from the match
                skills_in_match = re.findall(r'\b[a-zA-Z][a-zA-Z\s.+-]+\b', match)
                found_skills.extend([s.strip() for s in skills_in_match if len(s.strip()) > 2])
        
        return list(set(found_skills))[:10]  # Limit to 10 skills
    
    def _extract_experience_years(self, text: str) -> int:
        """Extract minimum experience years from text."""
        import re
        
        patterns = [
            r'(\d+)\+?\s*years?\s*(?:of\s*)?experience',
            r'minimum\s*(\d+)\s*years?',
            r'at least\s*(\d+)\s*years?'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text.lower())
            if match:
                return int(match.group(1))
        
        return 0
    
    def _extract_keywords(self, text: str) -> List[str]:
        """Extract industry keywords from text."""
        import re
        
        # Common industry terms
        industry_terms = [
            'software development', 'web development', 'data science',
            'machine learning', 'artificial intelligence', 'cloud computing',
            'devops', 'cybersecurity', 'mobile development', 'frontend',
            'backend', 'full stack', 'database', 'analytics', 'automation'
        ]
        
        found_keywords = []
        text_lower = text.lower()
        
        for term in industry_terms:
            if term in text_lower:
                found_keywords.append(term)
        
        return found_keywords[:8]  # Limit to 8 keywords
    
    def _review_and_edit_extracted_criteria(self, criteria_data: Dict[str, Any]) -> EvaluationCriteria:
        """Review and allow editing of extracted criteria."""
        console.print("\n[bold]Extracted Criteria Review[/bold]")
        
        # Show extracted criteria
        table = Table(title="Extracted Evaluation Criteria")
        table.add_column("Category", style="cyan")
        table.add_column("Extracted Values", style="white")
        
        table.add_row("Required Skills", ', '.join(criteria_data['required_skills']) or "None found")
        table.add_row("Preferred Skills", ', '.join(criteria_data['preferred_skills']) or "None found")
        table.add_row("Min Experience", f"{criteria_data['min_experience_years']} years")
        table.add_row("Industry Keywords", ', '.join(criteria_data['industry_keywords']) or "None found")
        
        console.print(table)
        
        if Confirm.ask("\nDo you want to edit these criteria?", default=False):
            # Allow interactive editing
            builder = InteractiveCriteriaBuilder()
            builder.criteria_data = criteria_data
            return builder.build_criteria_interactively()
        else:
            return EvaluationCriteria(**criteria_data)
