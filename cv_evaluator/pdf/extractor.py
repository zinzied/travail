"""
PDF text extraction utilities.
"""

import logging
from pathlib import Path
from typing import Optional, Dict, Any
import PyPDF2
import pdfplumber
import fitz  # PyMuPDF
from ..utils.exceptions import PDFExtractionError

logger = logging.getLogger(__name__)


class PDFExtractor:
    """Extract text and metadata from PDF files using multiple methods."""
    
    def __init__(self):
        self.extraction_methods = [
            self._extract_with_pdfplumber,
            self._extract_with_pymupdf,
            self._extract_with_pypdf2
        ]
    
    def extract_text(self, pdf_path: str) -> Dict[str, Any]:
        """
        Extract text from PDF using the best available method.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            Dictionary containing extracted text and metadata
        """
        pdf_path = Path(pdf_path)
        
        if not pdf_path.exists():
            raise PDFExtractionError(f"PDF file not found: {pdf_path}")
        
        if not pdf_path.suffix.lower() == '.pdf':
            raise PDFExtractionError(f"File is not a PDF: {pdf_path}")
        
        # Try extraction methods in order of preference
        for method in self.extraction_methods:
            try:
                result = method(pdf_path)
                if result and result.get('text', '').strip():
                    logger.info(f"Successfully extracted text using {method.__name__}")
                    return result
            except Exception as e:
                logger.warning(f"Failed to extract with {method.__name__}: {e}")
                continue
        
        raise PDFExtractionError(f"Failed to extract text from PDF: {pdf_path}")
    
    def _extract_with_pdfplumber(self, pdf_path: Path) -> Dict[str, Any]:
        """Extract text using pdfplumber (best for structured documents)."""
        text_content = []
        metadata = {}
        
        with pdfplumber.open(pdf_path) as pdf:
            metadata = {
                'pages': len(pdf.pages),
                'method': 'pdfplumber',
                'file_size': pdf_path.stat().st_size
            }
            
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text_content.append(page_text)
        
        return {
            'text': '\n'.join(text_content),
            'metadata': metadata,
            'confidence': 0.9 if text_content else 0.0
        }
    
    def _extract_with_pymupdf(self, pdf_path: Path) -> Dict[str, Any]:
        """Extract text using PyMuPDF (good for complex layouts)."""
        text_content = []
        
        doc = fitz.open(pdf_path)
        metadata = {
            'pages': doc.page_count,
            'method': 'pymupdf',
            'file_size': pdf_path.stat().st_size,
            'pdf_metadata': doc.metadata
        }
        
        for page_num in range(doc.page_count):
            page = doc[page_num]
            page_text = page.get_text()
            if page_text:
                text_content.append(page_text)
        
        doc.close()
        
        return {
            'text': '\n'.join(text_content),
            'metadata': metadata,
            'confidence': 0.8 if text_content else 0.0
        }
    
    def _extract_with_pypdf2(self, pdf_path: Path) -> Dict[str, Any]:
        """Extract text using PyPDF2 (fallback method)."""
        text_content = []
        
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            
            metadata = {
                'pages': len(pdf_reader.pages),
                'method': 'pypdf2',
                'file_size': pdf_path.stat().st_size
            }
            
            for page in pdf_reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text_content.append(page_text)
        
        return {
            'text': '\n'.join(text_content),
            'metadata': metadata,
            'confidence': 0.7 if text_content else 0.0
        }
    
    def validate_pdf(self, pdf_path: str) -> bool:
        """
        Validate if the file is a readable PDF.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            True if valid PDF, False otherwise
        """
        try:
            result = self.extract_text(pdf_path)
            return bool(result.get('text', '').strip())
        except Exception:
            return False
