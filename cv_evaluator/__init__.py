"""
CV Evaluator - AI-powered CV analysis and evaluation system.

Created by: Zied Boughdir (@zinzied)
GitHub: https://github.com/zinzied/cv-evaluation-system

An intelligent solution for analyzing and evaluating PDF CVs/resumes
with automated scoring and detailed report generation.
"""

__version__ = "1.0.0"
__author__ = "Zied Boughdir"
__email__ = "zied.boughdir@example.com"
__github__ = "https://github.com/zinzied"

from .core.evaluator import CVEvaluator
from .core.models import CVAnalysisResult, EvaluationCriteria
from .pdf.extractor import PDFExtractor
from .ai.analyzer import CVAnalyzer
from .reports.generator import ReportGenerator

__all__ = [
    "CVEvaluator",
    "CVAnalysisResult", 
    "EvaluationCriteria",
    "PDFExtractor",
    "CVAnalyzer",
    "ReportGenerator"
]
