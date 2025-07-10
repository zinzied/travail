"""
Setup script for CV Evaluator package.
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README file
readme_file = Path(__file__).parent / "README.md"
long_description = readme_file.read_text(encoding="utf-8") if readme_file.exists() else ""

# Read requirements
requirements_file = Path(__file__).parent / "requirements.txt"
requirements = []
if requirements_file.exists():
    requirements = requirements_file.read_text().strip().split('\n')
    requirements = [req.strip() for req in requirements if req.strip() and not req.startswith('#')]

setup(
    name="cv-evaluator",
    version="1.0.0",
    author="Zied Boughdir",
    author_email="zied.boughdir@example.com",
    description="AI-powered CV evaluation and analysis system",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/zinzied/cv-evaluation-system",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: End Users/Desktop",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Office/Business",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Text Processing :: Linguistic",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=22.0.0",
            "flake8>=5.0.0",
            "mypy>=1.0.0",
        ],
        "web": [
            "streamlit>=1.29.0",
            "fastapi>=0.104.1",
            "uvicorn>=0.24.0",
        ]
    },
    entry_points={
        "console_scripts": [
            "cv-evaluator=cv_evaluator.cli:app",
        ],
    },
    include_package_data=True,
    package_data={
        "cv_evaluator": [
            "config/*.yaml",
            "templates/*.html",
            "templates/*.jinja2",
        ],
    },
    zip_safe=False,
)
