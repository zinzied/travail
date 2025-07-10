# CV Evaluation System

An intelligent AI-powered solution for analyzing and evaluating PDF CVs/resumes with automated scoring and detailed report generation.

**Created by:** [Zied Boughdir](https://github.com/zinzied)  
**GitHub:** [@zinzied](https://github.com/zinzied)

## Features

This system was designed to streamline the CV evaluation process for HR professionals and recruiters:

- **Smart PDF Processing**: Intelligently extracts text and structured data from PDF CVs using multiple extraction methods
- **AI-Powered Analysis**: Uses advanced NLP to evaluate skills, experience, and qualifications with human-like understanding
- **Comprehensive Scoring**: Provides numerical ratings and detailed feedback based on customizable criteria
- **Professional Reports**: Generates detailed evaluation reports (PV - Procès-Verbal) that save hours of manual work
- **Efficient Batch Processing**: Process multiple CVs simultaneously with real-time progress tracking
- **Flexible Output**: Export results in PDF, Word, HTML, and JSON formats for different use cases
- **Industry Templates**: Pre-configured evaluation criteria for software engineers, data scientists, project managers, and more
- **Fully Customizable**: Adapt the system to your specific hiring needs and company requirements

## Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Quick Installation
```bash
# Clone the repository
git clone https://github.com/zinzied/cv-evaluation-system.git
cd cv-evaluation-system

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install the package
pip install -e .
```

### Automated Installation
```bash
# Run the installation script
python install.py
```

## Quick Start

### Web Interface (Recommended for beginners)
```bash
# Start the web application
python run_web_app.py
# Opens at http://localhost:8501
```

### Python API
```python
from cv_evaluator import CVEvaluator

# Create evaluator
evaluator = CVEvaluator()

# Evaluate a CV
result = evaluator.evaluate_cv("path/to/cv.pdf")

# Access results
print(f"Overall Score: {result.overall_score:.1f}/100")
print(f"Job Fit: {result.fit_percentage:.1f}%")
print(f"Candidate: {result.cv_data.personal_info.name}")

# Generate report
report_path = evaluator.generate_report(result, "report.pdf")
```

### Command Line Interface
```bash
# Evaluate a single CV
python -m cv_evaluator evaluate cv.pdf

# Batch process multiple CVs
python -m cv_evaluator batch input_folder/ output_folder/

# Use specific job template
python -m cv_evaluator evaluate cv.pdf --job-template software_engineer
```

## Usage Examples

### Single CV Evaluation
```python
from cv_evaluator import CVEvaluator

evaluator = CVEvaluator()
result = evaluator.evaluate_cv("john_doe_cv.pdf")

# View detailed results
print(f"Candidate: {result.cv_data.personal_info.name}")
print(f"Email: {result.cv_data.personal_info.email}")
print(f"Overall Score: {result.overall_score}/100")

# Section breakdown
for score in result.section_scores:
    print(f"{score.section}: {score.score}/{score.max_score}")

# Insights
print("Strengths:", result.strengths)
print("Areas for improvement:", result.weaknesses)
print("Recommendations:", result.recommendations)
```

### Batch Processing
```python
from cv_evaluator.core.batch_processor import BatchProcessor

processor = BatchProcessor(max_workers=5)
results = processor.process_directory(
    input_dir="cvs/",
    output_dir="results/",
    generate_reports=True,
    report_format="pdf"
)

print(f"Processed {results['successful']}/{results['total_files']} CVs")
```

### Custom Evaluation Criteria
```python
from cv_evaluator import CVEvaluator, EvaluationCriteria

# Define custom criteria for a data scientist role
criteria = EvaluationCriteria(
    required_skills=["python", "sql", "machine learning", "statistics"],
    preferred_skills=["tensorflow", "pandas", "aws"],
    min_experience_years=3,
    scoring_weights={
        "skills": 0.5,
        "experience": 0.3,
        "education": 0.15,
        "additional": 0.05
    }
)

evaluator = CVEvaluator(evaluation_criteria=criteria)
result = evaluator.evaluate_cv("data_scientist_cv.pdf")
```

## Configuration

### Job Templates
Use pre-configured templates for common positions:
```python
# Available templates: software_engineer, data_scientist, project_manager
evaluator = CVEvaluator(job_template="software_engineer")
```

### Custom Configuration
Edit `config/evaluation_criteria.yaml` to customize:
- Required and preferred skills
- Scoring weights
- Experience requirements
- Industry keywords

## Output Formats

The system generates comprehensive reports in multiple formats:

1. **PDF Reports**: Professional formatted reports with charts and detailed analysis
2. **HTML Reports**: Interactive web-friendly reports with visualizations
3. **Word Documents**: Editable reports for further customization
4. **JSON Data**: Structured data for integration with other systems

## Project Structure

```
cv-evaluation-system/
├── cv_evaluator/              # Main package
│   ├── core/                  # Core evaluation logic
│   ├── pdf/                   # PDF processing
│   ├── ai/                    # AI analysis modules
│   ├── reports/               # Report generation
│   ├── utils/                 # Utilities
│   ├── cli.py                 # Command-line interface
│   └── web_app.py             # Web interface
├── config/                    # Configuration files
├── examples/                  # Examples and demos
├── tests/                     # Test suite
├── requirements.txt           # Dependencies
└── README.md                  # This file
```

## Testing

Run the test suite to validate the system:
```bash
# Run all tests
pytest

# Run system validation
python test_system.py

# Run demo
python examples/demo.py
```

## Performance

- **Single CV**: 2-3 seconds processing time
- **Batch Processing**: Configurable concurrency (up to 8 workers tested)
- **Accuracy**: 70-95% text extraction confidence (depends on PDF quality)
- **Memory Efficient**: Handles large batches with proper resource management

## Troubleshooting

### Common Issues

1. **PDF Extraction Problems**: Ensure PDFs are text-based, not scanned images
2. **Low Extraction Confidence**: Check PDF formatting and quality
3. **Memory Issues**: Reduce batch size or number of workers
4. **Import Errors**: Verify all dependencies are installed

### Getting Help
- Check the logs: `logs/cv_evaluator.log`
- Run with verbose output: `--verbose` flag
- Validate files: `python -m cv_evaluator validate cv.pdf`
- Test installation: `python test_system.py`

## About the Author

**Zied Boughdir** is a passionate software engineer and AI enthusiast who created this system to help organizations make better hiring decisions through intelligent automation.

- **GitHub**: [@zinzied](https://github.com/zinzied)
- **LinkedIn**: Connect with Zied for collaboration opportunities
- **Email**: Contact through GitHub for project-related inquiries

## Acknowledgments

This project leverages several excellent open-source libraries:
- **ReportLab** for professional PDF generation
- **spaCy** for advanced natural language processing
- **scikit-learn** for machine learning utilities
- **Streamlit** for the intuitive web interface
- **Typer** for the command-line interface
- **Rich** for beautiful terminal output

## Support & Contributing

- **Issues**: [Report bugs or request features](https://github.com/zinzied/cv-evaluation-system/issues)
- **Discussions**: [Join the community](https://github.com/zinzied/cv-evaluation-system/discussions)
- **Contributing**: Pull requests are welcome! Please read the contributing guidelines first.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

**Made with ❤️ by [Zied Boughdir](https://github.com/zinzied) for smarter hiring decisions**
