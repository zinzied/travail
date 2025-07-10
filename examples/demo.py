#!/usr/bin/env python3
"""
Demo script showing how to use the CV Evaluator system.

Created by: Zied Boughdir (@zinzied)
GitHub: https://github.com/zinzied/cv-evaluation-system
"""

import sys
from pathlib import Path

# Add the parent directory to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from cv_evaluator.core.evaluator import CVEvaluator
from cv_evaluator.core.models import EvaluationCriteria
from cv_evaluator.core.criteria_loader import criteria_manager
from cv_evaluator.utils.logging_config import setup_logging


def demo_basic_evaluation():
    """Demonstrate basic CV evaluation."""
    print("=== Basic CV Evaluation Demo ===")
    
    # Setup logging
    setup_logging(level="INFO")
    
    # Create evaluator with default criteria
    evaluator = CVEvaluator()
    
    # For demo purposes, we'll create a mock CV text since we don't have a PDF
    sample_cv_text = """
    John Doe
    john.doe@example.com
    +1-555-0123
    
    EXPERIENCE
    Senior Software Engineer at Tech Corp
    2020 - Present
    • Developed web applications using Python and React
    • Led a team of 5 developers
    • Implemented CI/CD pipelines
    
    EDUCATION
    Master of Science in Computer Science
    MIT, 2018
    
    SKILLS
    Programming: Python, Java, JavaScript, SQL
    Web: React, Node.js, HTML, CSS
    Databases: PostgreSQL, MySQL, MongoDB
    Cloud: AWS, Docker, Kubernetes
    Tools: Git, Jenkins, Terraform
    
    LANGUAGES
    English, Spanish
    
    CERTIFICATIONS
    AWS Certified Solutions Architect
    Certified Scrum Master
    """
    
    # Parse CV directly (simulating PDF extraction)
    cv_data = evaluator.cv_parser.parse_cv(sample_cv_text)
    
    # Analyze CV
    analysis_result = evaluator.analyzer.analyze_cv(cv_data)
    
    # Display results
    print(f"Candidate: {analysis_result.cv_data.personal_info.name}")
    print(f"Overall Score: {analysis_result.overall_score:.1f}/100")
    print(f"Job Fit: {analysis_result.fit_percentage:.1f}%")
    
    print("\nSection Scores:")
    for score in analysis_result.section_scores:
        percentage = (score.score / score.max_score) * 100
        print(f"  {score.section.title()}: {score.score:.1f}/{score.max_score:.1f} ({percentage:.1f}%)")
    
    print(f"\nStrengths ({len(analysis_result.strengths)}):")
    for strength in analysis_result.strengths:
        print(f"  • {strength}")
    
    print(f"\nAreas for Improvement ({len(analysis_result.weaknesses)}):")
    for weakness in analysis_result.weaknesses:
        print(f"  • {weakness}")
    
    print(f"\nRecommendations ({len(analysis_result.recommendations)}):")
    for recommendation in analysis_result.recommendations:
        print(f"  • {recommendation}")
    
    return analysis_result


def demo_custom_criteria():
    """Demonstrate evaluation with custom criteria."""
    print("\n=== Custom Criteria Demo ===")
    
    # Create custom evaluation criteria for a data scientist position
    custom_criteria = EvaluationCriteria(
        required_skills=["python", "sql", "machine learning", "statistics"],
        preferred_skills=["pandas", "numpy", "scikit-learn", "tensorflow", "tableau"],
        min_experience_years=3,
        industry_keywords=["data science", "analytics", "machine learning", "statistics"],
        scoring_weights={
            "skills": 0.5,      # Higher weight on skills for technical role
            "experience": 0.3,
            "education": 0.15,
            "additional": 0.05
        }
    )
    
    # Create evaluator with custom criteria
    evaluator = CVEvaluator(evaluation_criteria=custom_criteria)
    
    # Sample data scientist CV
    data_scientist_cv = """
    Jane Smith
    jane.smith@example.com
    
    EXPERIENCE
    Senior Data Scientist at DataCorp
    2019 - Present
    • Built machine learning models for customer segmentation
    • Analyzed large datasets using Python and SQL
    • Created data visualizations with Tableau
    
    Data Analyst at Analytics Inc
    2017 - 2019
    • Performed statistical analysis on business data
    • Developed predictive models using scikit-learn
    
    EDUCATION
    PhD in Statistics
    Stanford University, 2017
    
    SKILLS
    Programming: Python, R, SQL
    ML Libraries: scikit-learn, TensorFlow, pandas, numpy
    Visualization: Tableau, matplotlib, seaborn
    Statistics: Hypothesis testing, regression analysis
    
    CERTIFICATIONS
    Google Cloud Professional Data Engineer
    Tableau Desktop Specialist
    """
    
    # Parse and analyze
    cv_data = evaluator.cv_parser.parse_cv(data_scientist_cv)
    analysis_result = evaluator.analyzer.analyze_cv(cv_data)
    
    print(f"Data Scientist Evaluation:")
    print(f"Overall Score: {analysis_result.overall_score:.1f}/100")
    print(f"Job Fit: {analysis_result.fit_percentage:.1f}%")
    
    # Show skills analysis
    print(f"\nSkills Found ({len(analysis_result.cv_data.skills)}):")
    for skill in analysis_result.cv_data.skills[:10]:  # Show top 10
        print(f"  • {skill.name} ({skill.category or 'General'}) - Confidence: {skill.confidence:.2f}")
    
    return analysis_result


def demo_job_templates():
    """Demonstrate using predefined job templates."""
    print("\n=== Job Templates Demo ===")
    
    # List available templates
    templates = criteria_manager.list_job_templates()
    print(f"Available job templates: {templates}")
    
    if "software_engineer" in templates:
        # Use software engineer template
        evaluator = CVEvaluator(job_template="software_engineer")
        
        print("\nUsing 'software_engineer' template:")
        criteria = evaluator.get_evaluation_criteria()
        print(f"Required skills: {criteria.required_skills}")
        print(f"Preferred skills: {criteria.preferred_skills}")
        print(f"Min experience: {criteria.min_experience_years} years")
    
    if "data_scientist" in templates:
        # Use data scientist template
        evaluator = CVEvaluator(job_template="data_scientist")
        
        print("\nUsing 'data_scientist' template:")
        criteria = evaluator.get_evaluation_criteria()
        print(f"Required skills: {criteria.required_skills}")
        print(f"Preferred skills: {criteria.preferred_skills}")
        print(f"Min experience: {criteria.min_experience_years} years")


def demo_report_generation():
    """Demonstrate report generation."""
    print("\n=== Report Generation Demo ===")
    
    # Use the result from basic evaluation
    evaluator = CVEvaluator()
    
    sample_cv_text = """
    Alice Johnson
    alice.johnson@example.com
    
    EXPERIENCE
    Full Stack Developer at WebCorp
    2021 - Present
    • Developed React applications
    • Built REST APIs with Node.js
    
    SKILLS
    JavaScript, React, Node.js, Python, SQL
    
    EDUCATION
    BS Computer Science, UC Berkeley, 2021
    """
    
    cv_data = evaluator.cv_parser.parse_cv(sample_cv_text)
    analysis_result = evaluator.analyzer.analyze_cv(cv_data)
    
    # Generate HTML report (since it doesn't require external dependencies)
    try:
        report_path = evaluator.generate_report(
            analysis_result,
            "examples/sample_report.html",
            format="html"
        )
        print(f"HTML report generated: {report_path}")
    except Exception as e:
        print(f"Report generation failed: {e}")
        print("This is expected if templates are not properly configured.")


def main():
    """Run all demos."""
    print("CV Evaluator System Demo")
    print("=" * 50)
    
    try:
        # Run demos
        demo_basic_evaluation()
        demo_custom_criteria()
        demo_job_templates()
        demo_report_generation()
        
        print("\n" + "=" * 50)
        print("Demo completed successfully!")
        print("\nTo use the system:")
        print("1. Install dependencies: pip install -r requirements.txt")
        print("2. Use CLI: python -m cv_evaluator evaluate path/to/cv.pdf")
        print("3. Or use Python API as shown in this demo")
        
    except Exception as e:
        print(f"\nDemo failed with error: {e}")
        print("This might be due to missing dependencies or configuration.")
        print("Please ensure all requirements are installed.")


if __name__ == "__main__":
    main()
