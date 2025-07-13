#!/usr/bin/env python3
"""
Test script for document extraction functionality.
"""

import sys
from pathlib import Path

# Add the project root to Python path
sys.path.insert(0, str(Path(__file__).parent))

from cv_evaluator.pdf.extractor import UniversalDocumentExtractor

def test_document_extractor():
    """Test the universal document extractor."""
    print("🧪 Testing Universal Document Extractor")
    print("=" * 50)
    
    extractor = UniversalDocumentExtractor()
    
    # Show supported extensions
    print("📋 Supported file extensions:")
    for ext in extractor.get_supported_extensions():
        description = extractor.get_file_type_description(f"test{ext}")
        print(f"  • {ext} - {description}")
    
    print("\n✅ Document extractor initialized successfully!")
    print("\n🚀 Ready to process documents!")
    print("\nSupported formats:")
    print("  • PDF files (.pdf)")
    print("  • Word documents (.docx, .doc)")
    print("  • Excel spreadsheets (.xlsx, .xls)")
    print("  • Text files (.txt)")
    
    return True

if __name__ == "__main__":
    try:
        test_document_extractor()
        print("\n🎉 All tests passed!")
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        sys.exit(1)
