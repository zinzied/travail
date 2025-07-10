# CV Evaluator - Complete Usage Guide

This guide provides comprehensive instructions for using the CV Evaluation System.

## 🚀 Quick Installation

### Option 1: Automated Installation (Recommended)
```bash
# Clone the repository
git clone <repository-url>
cd cv-evaluation-system

# Run the installation script
python install.py
```

### Option 2: Manual Installation
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install the package
pip install -e .

# Set up environment
cp .env.example .env
```

## 💻 Command Line Usage

### Single CV Evaluation

#### Basic Evaluation
```bash
# Evaluate a CV with default criteria
python -m cv_evaluator evaluate cv.pdf

# Specify output file and format
python -m cv_evaluator evaluate cv.pdf --output report.pdf --format pdf
```

#### Using Job Templates
```bash
# List available job templates
python -m cv_evaluator list-templates

# Use specific job template
python -m cv_evaluator evaluate cv.pdf --job-template software_engineer
python -m cv_evaluator evaluate cv.pdf --job-template data_scientist
```

#### Advanced Options
```bash
# Verbose output for debugging
python -m cv_evaluator evaluate cv.pdf --verbose

# Custom criteria configuration
python -m cv_evaluator evaluate cv.pdf --criteria custom_criteria

# Generate HTML report
python -m cv_evaluator evaluate cv.pdf --format html --output report.html
```

### Batch Processing

#### Process Directory
```bash
# Process all PDFs in a directory
python -m cv_evaluator batch input_folder/ output_folder/

# Custom file pattern
python -m cv_evaluator batch input_folder/ output_folder/ --pattern "resume_*.pdf"

# Skip individual reports (summary only)
python -m cv_evaluator batch input_folder/ output_folder/ --no-reports
```

#### Performance Tuning
```bash
# Control concurrent workers
python -m cv_evaluator batch input_folder/ output_folder/ --workers 5

# Generate Word documents instead of PDFs
python -m cv_evaluator batch input_folder/ output_folder/ --format word
```

### Utility Commands

```bash
# Validate a CV file
python -m cv_evaluator validate cv.pdf

# Show current configuration
python -m cv_evaluator config-info

# List available evaluation criteria
python -m cv_evaluator list-criteria
```

## 🐍 Python API Usage

### Basic API Usage

```python
from cv_evaluator import CVEvaluator

# Create evaluator
evaluator = CVEvaluator()

# Evaluate CV
result = evaluator.evaluate_cv("path/to/cv.pdf")

# Access results
print(f"Overall Score: {result.overall_score:.1f}/100")
print(f"Job Fit: {result.fit_percentage:.1f}%")
print(f"Candidate: {result.cv_data.personal_info.name}")

# Generate report
report_path = evaluator.generate_report(result, "report.pdf")
```

### Advanced API Usage

#### Custom Evaluation Criteria
```python
from cv_evaluator import CVEvaluator, EvaluationCriteria

# Define custom criteria
criteria = EvaluationCriteria(
    required_skills=["python", "sql", "machine learning"],
    preferred_skills=["tensorflow", "aws", "docker"],
    min_experience_years=3,
    scoring_weights={
        "skills": 0.5,
        "experience": 0.3,
        "education": 0.15,
        "additional": 0.05
    }
)

# Create evaluator with custom criteria
evaluator = CVEvaluator(evaluation_criteria=criteria)
```

#### Batch Processing API
```python
from cv_evaluator.core.batch_processor import BatchProcessor

# Create batch processor
processor = BatchProcessor(max_workers=5)

# Process multiple files
results = processor.process_file_list(
    file_paths=["cv1.pdf", "cv2.pdf", "cv3.pdf"],
    output_dir="results/",
    generate_reports=True,
    report_format="pdf"
)

# Access results
print(f"Processed: {results['successful']}/{results['total_files']}")
for result in results['results']:
    if result['success']:
        print(f"- {result['candidate_name']}: {result['overall_score']:.1f}")
```

#### Working with Results
```python
# Detailed result analysis
result = evaluator.evaluate_cv("cv.pdf")

# Personal information
personal = result.cv_data.personal_info
print(f"Name: {personal.name}")
print(f"Email: {personal.email}")
print(f"Phone: {personal.phone}")

# Skills analysis
for skill in result.cv_data.skills:
    print(f"Skill: {skill.name} ({skill.category}) - {skill.confidence:.2f}")

# Section scores
for score in result.section_scores:
    percentage = (score.score / score.max_score) * 100
    print(f"{score.section}: {percentage:.1f}% - {score.feedback}")

# Insights
print("Strengths:", result.strengths)
print("Weaknesses:", result.weaknesses)
print("Recommendations:", result.recommendations)
```

## 🌐 Web Interface Usage

### Starting the Web Application
```bash
# Start the web interface
python run_web_app.py

# Or manually with Streamlit
streamlit run cv_evaluator/web_app.py
```

### Web Interface Features

1. **Single CV Evaluation**
   - Upload PDF file
   - Select job template
   - Choose report format
   - View interactive results
   - Download generated report

2. **Batch Processing**
   - Upload multiple PDF files
   - Configure processing options
   - Monitor progress
   - View summary statistics
   - Download results

3. **Interactive Visualizations**
   - Section score charts
   - Skills analysis
   - Score distributions
   - Candidate comparisons

## ⚙️ Configuration

### Environment Configuration (.env)
```bash
# Application settings
APP_NAME=CV Evaluation System
DEBUG=False

# Evaluation settings
SCORING_SCALE=100
MIN_SCORE_THRESHOLD=50

# File processing
MAX_FILE_SIZE_MB=10
TEMP_DIR=temp/

# Batch processing
MAX_CONCURRENT_JOBS=5
BATCH_TIMEOUT=300

# Logging
LOG_LEVEL=INFO
LOG_FILE=logs/cv_evaluator.log
```

### Evaluation Criteria Configuration

Edit `config/evaluation_criteria.yaml`:

```yaml
# Scoring weights
scoring_weights:
  skills: 0.4
  experience: 0.3
  education: 0.2
  additional: 0.1

# Required skills
skills:
  required:
    - "communication"
    - "problem solving"
  preferred:
    - "leadership"
    - "project management"

# Experience requirements
experience:
  min_years: 2
  industry_keywords:
    - "software development"
    - "data analysis"
```

### Job Templates

Create custom job templates in the configuration:

```yaml
job_templates:
  senior_developer:
    required_skills:
      - "programming"
      - "system design"
      - "leadership"
    min_experience_years: 5
    
  data_analyst:
    required_skills:
      - "sql"
      - "data analysis"
      - "statistics"
    preferred_skills:
      - "python"
      - "tableau"
    min_experience_years: 2
```

## 📊 Understanding Results

### Scoring System

The system uses a weighted scoring approach:

- **Skills (40%)**: Technical and soft skills evaluation
- **Experience (30%)**: Work history and career progression
- **Education (20%)**: Educational background and qualifications
- **Additional (10%)**: Languages, certifications, projects

### Score Interpretation

- **85-100**: Excellent candidate, strong match
- **70-84**: Good candidate, meets most requirements
- **55-69**: Average candidate, some gaps
- **40-54**: Below average, significant improvements needed
- **0-39**: Poor match, major gaps in requirements

### Report Sections

1. **Executive Summary**: Overall scores and key metrics
2. **Section Analysis**: Detailed breakdown by category
3. **Skills Assessment**: Technical and soft skills evaluation
4. **Experience Review**: Work history analysis
5. **Education Evaluation**: Academic background assessment
6. **Recommendations**: Actionable improvement suggestions

## 🔧 Troubleshooting

### Common Issues

#### PDF Extraction Problems
```python
# Check if PDF is valid
evaluator = CVEvaluator()
if not evaluator.validate_cv_file("cv.pdf"):
    print("PDF file may be corrupted or password-protected")

# Check extraction confidence
result = evaluator.evaluate_cv("cv.pdf")
if result.cv_data.extraction_confidence < 0.7:
    print("Low extraction quality - consider manual review")
```

#### Memory Issues with Large Batches
```python
# Process in smaller chunks
def process_large_batch(files, chunk_size=20):
    for i in range(0, len(files), chunk_size):
        chunk = files[i:i + chunk_size]
        results = processor.process_file_list(chunk, "output/")
        # Process results...
```

#### Configuration Errors
```bash
# Validate configuration
python -c "from cv_evaluator.core.criteria_loader import criteria_manager; print(criteria_manager.get_criteria())"
```

### Performance Optimization

1. **Adjust Worker Count**: Match your CPU cores
2. **Use SSD Storage**: For faster file I/O
3. **Increase Memory**: For large batch processing
4. **Enable Caching**: For repeated evaluations

### Getting Help

- Check the logs: `logs/cv_evaluator.log`
- Run with verbose output: `--verbose` flag
- Validate files: `python -m cv_evaluator validate cv.pdf`
- Test installation: `python examples/demo.py`

## 📚 Examples

See the `examples/` directory for:
- `demo.py`: Complete API demonstration
- `sample_cv.txt`: Example CV content
- Sample configuration files
- Integration examples

## 🤝 Support

For additional help:
- Check the README.md for detailed documentation
- Review the examples directory
- Run the demo script for hands-on learning
- Use the `--help` flag with CLI commands
