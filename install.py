#!/usr/bin/env python3
"""
Installation script for CV Evaluator system.

Created by: Zied Boughdir (@zinzied)
GitHub: https://github.com/zinzied/cv-evaluation-system
"""

import sys
import subprocess
import os
from pathlib import Path
import platform

def check_python_version():
    """Check if Python version is compatible."""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Python 3.8 or higher is required")
        print(f"Current version: {version.major}.{version.minor}.{version.micro}")
        return False
    
    print(f"✅ Python {version.major}.{version.minor}.{version.micro} detected")
    return True

def check_pip():
    """Check if pip is available."""
    try:
        import pip
        print("✅ pip is available")
        return True
    except ImportError:
        print("❌ pip is not available")
        return False

def install_requirements():
    """Install required packages."""
    print("📦 Installing required packages...")
    
    requirements_file = Path("requirements.txt")
    if not requirements_file.exists():
        print("❌ requirements.txt not found")
        return False
    
    try:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "-r", "requirements.txt"
        ])
        print("✅ Requirements installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install requirements: {e}")
        return False

def install_optional_dependencies():
    """Install optional dependencies for enhanced features."""
    print("🔧 Installing optional dependencies...")
    
    optional_packages = [
        "streamlit",  # For web interface
        "plotly",     # For interactive charts
        "pytest",     # For testing
        "black",      # For code formatting
    ]
    
    for package in optional_packages:
        try:
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", package
            ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            print(f"✅ {package} installed")
        except subprocess.CalledProcessError:
            print(f"⚠️  Failed to install {package} (optional)")

def download_nlp_models():
    """Download required NLP models."""
    print("🧠 Downloading NLP models...")
    
    models = ["en_core_web_sm"]
    
    for model in models:
        try:
            subprocess.check_call([
                sys.executable, "-m", "spacy", "download", model
            ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            print(f"✅ {model} downloaded")
        except subprocess.CalledProcessError:
            print(f"⚠️  Failed to download {model} (optional)")

def setup_environment():
    """Set up environment configuration."""
    print("⚙️  Setting up environment...")
    
    env_example = Path(".env.example")
    env_file = Path(".env")
    
    if env_example.exists() and not env_file.exists():
        try:
            env_file.write_text(env_example.read_text())
            print("✅ Environment file created (.env)")
        except Exception as e:
            print(f"⚠️  Failed to create .env file: {e}")
    
    # Create necessary directories
    directories = ["logs", "temp", "examples/output"]
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
    
    print("✅ Directories created")

def install_package():
    """Install the CV Evaluator package."""
    print("📦 Installing CV Evaluator package...")
    
    try:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "-e", "."
        ])
        print("✅ CV Evaluator package installed")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install package: {e}")
        return False

def run_tests():
    """Run basic tests to verify installation."""
    print("🧪 Running basic tests...")
    
    try:
        # Test import
        import cv_evaluator
        print("✅ Package import successful")
        
        # Test CLI
        result = subprocess.run([
            sys.executable, "-m", "cv_evaluator", "--help"
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ CLI interface working")
        else:
            print("⚠️  CLI interface may have issues")
        
        return True
    except Exception as e:
        print(f"⚠️  Test failed: {e}")
        return False

def print_usage_instructions():
    """Print usage instructions."""
    print("\n" + "="*60)
    print("🎉 Installation completed successfully!")
    print("="*60)
    
    print("\n📚 Quick Start:")
    print("1. Command Line Interface:")
    print("   python -m cv_evaluator evaluate path/to/cv.pdf")
    print("   python -m cv_evaluator batch input_folder/ output_folder/")
    
    print("\n2. Web Interface:")
    print("   python run_web_app.py")
    
    print("\n3. Python API:")
    print("   python examples/demo.py")
    
    print("\n📖 Documentation:")
    print("   - README.md for detailed documentation")
    print("   - examples/ folder for sample usage")
    print("   - config/ folder for configuration options")
    
    print("\n🔧 Configuration:")
    print("   - Edit .env file for environment settings")
    print("   - Edit config/evaluation_criteria.yaml for evaluation criteria")
    
    print("\n🧪 Testing:")
    print("   pytest                    # Run all tests")
    print("   python examples/demo.py   # Run demo script")
    
    print("\n❓ Need help?")
    print("   - Check README.md for troubleshooting")
    print("   - Run: python -m cv_evaluator --help")

def main():
    """Main installation process."""
    print("🚀 CV Evaluator Installation Script")
    print("Created by: Zied Boughdir (@zinzied)")
    print("="*50)
    
    # Check system requirements
    if not check_python_version():
        sys.exit(1)
    
    if not check_pip():
        sys.exit(1)
    
    print(f"💻 Platform: {platform.system()} {platform.release()}")
    print(f"📁 Installation directory: {Path.cwd()}")
    
    # Confirm installation
    response = input("\n🤔 Do you want to proceed with installation? (y/N): ")
    if response.lower() not in ['y', 'yes']:
        print("❌ Installation cancelled")
        sys.exit(0)
    
    print("\n🔄 Starting installation process...")
    
    # Installation steps
    steps = [
        ("Installing core requirements", install_requirements),
        ("Installing optional dependencies", install_optional_dependencies),
        ("Setting up environment", setup_environment),
        ("Installing package", install_package),
        ("Downloading NLP models", download_nlp_models),
        ("Running tests", run_tests),
    ]
    
    failed_steps = []
    
    for step_name, step_func in steps:
        print(f"\n📋 {step_name}...")
        try:
            if not step_func():
                failed_steps.append(step_name)
        except Exception as e:
            print(f"❌ {step_name} failed: {e}")
            failed_steps.append(step_name)
    
    # Summary
    print("\n" + "="*50)
    if failed_steps:
        print("⚠️  Installation completed with some issues:")
        for step in failed_steps:
            print(f"   - {step}")
        print("\nThe system should still work, but some features may be limited.")
    else:
        print("✅ Installation completed successfully!")
    
    print_usage_instructions()

if __name__ == "__main__":
    main()
