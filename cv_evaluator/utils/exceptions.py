"""
Custom exceptions for CV evaluation system.
"""


class CVEvaluatorError(Exception):
    """Base exception for CV evaluator."""
    pass


class PDFExtractionError(CVEvaluatorError):
    """Exception raised when PDF extraction fails."""
    pass


class AnalysisError(CVEvaluatorError):
    """Exception raised when CV analysis fails."""
    pass


class ConfigurationError(CVEvaluatorError):
    """Exception raised when configuration is invalid."""
    pass


class ReportGenerationError(CVEvaluatorError):
    """Exception raised when report generation fails."""
    pass
