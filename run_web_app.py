#!/usr/bin/env python3
"""
Script to run the CV Evaluator web application.
"""

import sys
import subprocess
import os
from pathlib import Path

def main():
    """Run the Streamlit web application."""
    
    # Add current directory to Python path
    current_dir = Path(__file__).parent
    sys.path.insert(0, str(current_dir))
    
    # Check if streamlit is installed
    try:
        import streamlit
    except ImportError:
        print("❌ Streamlit is not installed.")
        print("Please install it with: pip install streamlit")
        sys.exit(1)
    
    # Check if plotly is installed (for charts)
    try:
        import plotly
    except ImportError:
        print("⚠️  Plotly is not installed. Charts may not work properly.")
        print("Install it with: pip install plotly")
    
    # Run the Streamlit app
    app_path = current_dir / "cv_evaluator" / "web_app.py"
    
    if not app_path.exists():
        print(f"❌ Web app file not found: {app_path}")
        sys.exit(1)
    
    print("🚀 Starting CV Evaluator Web Application...")
    print("📱 The app will open in your default web browser")
    print("🛑 Press Ctrl+C to stop the application")
    print("-" * 50)
    
    # Run streamlit
    try:
        # Set environment variable to help with imports
        env = dict(os.environ)
        env['PYTHONPATH'] = str(current_dir)

        subprocess.run([
            sys.executable, "-m", "streamlit", "run", str(app_path),
            "--server.port", "8501",
            "--server.address", "localhost",
            "--browser.gatherUsageStats", "false"
        ], env=env)
    except KeyboardInterrupt:
        print("\n👋 Application stopped by user")
    except Exception as e:
        print(f"❌ Failed to start application: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
