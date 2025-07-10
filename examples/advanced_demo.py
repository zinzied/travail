#!/usr/bin/env python3
"""
Advanced demo showing new features: custom criteria and participant evaluation.

Created by: Zied Boughdir (@zinzied)
GitHub: https://github.com/zinzied/cv-evaluation-system
"""

import sys
import tempfile
from pathlib import Path

# Add parent directory to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from cv_evaluator.core.interactive_criteria import InteractiveCriteriaBuilder, CriteriaFromFiles
from cv_evaluator.core.participant_evaluator import ParticipantEvaluator
from cv_evaluator.core.models import EvaluationCriteria


def demo_custom_criteria_creation():
    """Demonstrate creating custom evaluation criteria programmatically."""
    print("=== Demo 1: Custom Criteria Creation ===")
    
    # Create custom criteria for a Data Scientist position
    custom_criteria = EvaluationCriteria(
        required_skills=[
            "python", "sql", "statistics", "machine learning", 
            "data analysis", "communication"
        ],
        preferred_skills=[
            "tensorflow", "pytorch", "pandas", "numpy", "scikit-learn",
            "tableau", "power bi", "aws", "docker", "git"
        ],
        min_experience_years=3,
        industry_keywords=[
            "data science", "machine learning", "analytics", 
            "statistical modeling", "data visualization"
        ],
        scoring_weights={
            "skills": 0.45,      # Higher weight on technical skills
            "experience": 0.35,   # Strong experience requirement
            "education": 0.15,    # Advanced degree preferred
            "additional": 0.05    # Certifications and projects
        },
        max_score=100
    )
    
    print("✅ Custom Data Scientist criteria created")
    print(f"   Required skills: {len(custom_criteria.required_skills)}")
    print(f"   Preferred skills: {len(custom_criteria.preferred_skills)}")
    print(f"   Min experience: {custom_criteria.min_experience_years} years")
    
    # Save criteria to file
    builder = InteractiveCriteriaBuilder()
    criteria_file = builder.save_criteria_to_file(custom_criteria, "data_scientist_demo")
    print(f"   Saved to: {criteria_file}")
    
    return custom_criteria


def demo_criteria_from_job_description():
    """Demonstrate extracting criteria from job description text."""
    print("\n=== Demo 2: Extract Criteria from Job Description ===")
    
    # Sample job description
    job_description = """
    Senior Data Scientist Position
    
    We are looking for an experienced Data Scientist to join our team.
    
    Required Skills:
    - Python programming (5+ years experience)
    - SQL and database management
    - Machine learning and statistical modeling
    - Data visualization tools (Tableau, Power BI)
    - Strong communication and presentation skills
    
    Preferred Skills:
    - TensorFlow or PyTorch experience
    - AWS cloud platform knowledge
    - Docker containerization
    - Git version control
    - PhD in Statistics, Computer Science, or related field
    
    Minimum 4 years of experience in data science or analytics roles.
    Experience with large-scale data processing and real-time analytics preferred.
    """
    
    # Save job description to temporary file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write(job_description)
        job_desc_file = f.name
    
    try:
        # Extract criteria from job description
        extractor = CriteriaFromFiles()
        extracted_criteria = extractor._analyze_text_for_criteria(job_description)
        
        print("✅ Criteria extracted from job description")
        print(f"   Required skills found: {extracted_criteria['required_skills']}")
        print(f"   Preferred skills found: {extracted_criteria['preferred_skills']}")
        print(f"   Min experience: {extracted_criteria['min_experience_years']} years")
        print(f"   Industry keywords: {extracted_criteria['industry_keywords']}")
        
        # Create EvaluationCriteria object
        criteria = EvaluationCriteria(**extracted_criteria)
        return criteria
        
    finally:
        # Clean up
        Path(job_desc_file).unlink(missing_ok=True)


def demo_participant_evaluation():
    """Demonstrate evaluating a participant with multiple files."""
    print("\n=== Demo 3: Participant Evaluation with Multiple Files ===")
    
    # Create sample participant files
    participant_files = {}
    
    # Sample CV
    cv_content = """
    Dr. Sarah Chen
    sarah.chen@email.com
    +1-555-0123
    LinkedIn: linkedin.com/in/sarahchen
    
    PROFESSIONAL SUMMARY
    Experienced Data Scientist with 6+ years in machine learning and analytics.
    PhD in Statistics with expertise in deep learning and big data processing.
    
    TECHNICAL SKILLS
    Programming: Python, R, SQL, Scala
    ML/AI: TensorFlow, PyTorch, scikit-learn, Keras
    Data Tools: Pandas, NumPy, Matplotlib, Seaborn
    Big Data: Spark, Hadoop, Kafka
    Cloud: AWS (S3, EC2, SageMaker), Azure
    Databases: PostgreSQL, MongoDB, Redis
    Visualization: Tableau, Power BI, D3.js
    
    PROFESSIONAL EXPERIENCE
    
    Senior Data Scientist | TechCorp Inc. | 2020 - Present
    • Lead ML model development for recommendation systems (10M+ users)
    • Implemented real-time analytics pipeline processing 1TB+ daily
    • Mentored team of 4 junior data scientists
    • Technologies: Python, TensorFlow, AWS, Spark
    
    Data Scientist | Analytics Solutions | 2018 - 2020
    • Developed predictive models for customer churn (95% accuracy)
    • Built automated reporting dashboards using Tableau
    • Collaborated with product teams on A/B testing
    • Technologies: Python, scikit-learn, SQL, Tableau
    
    EDUCATION
    PhD in Statistics | Stanford University | 2018
    Dissertation: "Deep Learning Approaches for Time Series Forecasting"
    
    MS in Computer Science | MIT | 2015
    BS in Mathematics | UC Berkeley | 2013
    
    CERTIFICATIONS
    • AWS Certified Machine Learning - Specialty (2022)
    • Google Cloud Professional Data Engineer (2021)
    • Tableau Desktop Specialist (2020)
    
    PUBLICATIONS
    • "Advanced Neural Networks for Financial Forecasting" - Nature Machine Intelligence (2022)
    • "Scalable ML Pipelines for Real-time Analytics" - ICML (2021)
    
    LANGUAGES
    English (Native), Mandarin (Fluent), Spanish (Intermediate)
    """
    
    # Sample cover letter
    cover_letter_content = """
    Dear Hiring Manager,
    
    I am excited to apply for the Senior Data Scientist position at your company.
    With over 6 years of experience in machine learning and data analytics, I have
    developed expertise in building scalable ML systems and leading data science teams.
    
    In my current role at TechCorp, I have successfully:
    - Led the development of recommendation algorithms serving 10M+ users
    - Implemented real-time data processing pipelines handling terabytes of data
    - Mentored junior team members and established ML best practices
    - Collaborated with cross-functional teams to deliver data-driven solutions
    
    My technical expertise includes Python, TensorFlow, AWS, and big data technologies.
    I hold a PhD in Statistics from Stanford and have published research in top-tier
    machine learning conferences.
    
    I am particularly interested in your company's focus on AI-driven innovation
    and would love to contribute to your data science initiatives.
    
    Best regards,
    Dr. Sarah Chen
    """
    
    # Sample portfolio description
    portfolio_content = """
    DATA SCIENCE PORTFOLIO - Dr. Sarah Chen
    
    PROJECT 1: Customer Churn Prediction System
    - Built ensemble model combining XGBoost and neural networks
    - Achieved 95% accuracy on customer churn prediction
    - Deployed model to production serving 1M+ predictions daily
    - Technologies: Python, scikit-learn, XGBoost, AWS Lambda
    
    PROJECT 2: Real-time Recommendation Engine
    - Developed collaborative filtering system for e-commerce platform
    - Implemented using TensorFlow and deployed on Kubernetes
    - Increased user engagement by 35% and revenue by 20%
    - Technologies: TensorFlow, Kubernetes, Redis, Apache Kafka
    
    PROJECT 3: Financial Time Series Forecasting
    - Created LSTM-based model for stock price prediction
    - Published research paper in Nature Machine Intelligence
    - Open-sourced implementation with 500+ GitHub stars
    - Technologies: PyTorch, LSTM, Financial APIs
    
    PROJECT 4: Computer Vision for Medical Imaging
    - Developed CNN model for medical image classification
    - Achieved 98% accuracy on skin cancer detection
    - Collaborated with medical professionals for validation
    - Technologies: PyTorch, OpenCV, Medical imaging datasets
    """
    
    # Create temporary files
    temp_files = []
    
    # CV file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write(cv_content)
        participant_files['cv'] = f.name
        temp_files.append(f.name)
    
    # Cover letter file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write(cover_letter_content)
        participant_files['cover_letter'] = f.name
        temp_files.append(f.name)
    
    # Portfolio file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write(portfolio_content)
        participant_files['portfolio'] = f.name
        temp_files.append(f.name)
    
    try:
        # Create custom criteria for evaluation
        criteria = EvaluationCriteria(
            required_skills=["python", "machine learning", "statistics", "sql"],
            preferred_skills=["tensorflow", "aws", "deep learning", "big data"],
            min_experience_years=5,
            scoring_weights={
                "skills": 0.4,
                "experience": 0.35,
                "education": 0.2,
                "additional": 0.05
            }
        )
        
        # Create participant evaluator
        evaluator = ParticipantEvaluator(criteria)
        
        # Add participant with multiple files
        participant_id = "CANDIDATE_001"
        files_list = [
            {'path': participant_files['cv'], 'type': 'cv', 'description': 'Main CV/Resume'},
            {'path': participant_files['cover_letter'], 'type': 'cover_letter', 'description': 'Cover letter'},
            {'path': participant_files['portfolio'], 'type': 'portfolio', 'description': 'Project portfolio'}
        ]
        
        evaluator.add_participant_files(participant_id, files_list)
        
        print(f"✅ Added participant: {participant_id}")
        print(f"   Files: {len(files_list)}")
        
        # Process participant files
        print("🔄 Processing participant files...")
        success = evaluator.process_participant_files(participant_id)
        
        if success:
            print("✅ Files processed successfully")
            
            # Show file processing status
            participant = evaluator.participants[participant_id]
            for file_obj in participant.files:
                status = "✅" if file_obj.processing_status == "completed" else "❌"
                print(f"   {status} {file_obj.file_type}: {file_obj.processing_status}")
        
        # Evaluate participant
        print("🔄 Evaluating participant...")
        result = evaluator.evaluate_participant(participant_id)
        
        if result:
            print("✅ Evaluation completed!")
            print(f"   Candidate: {result.cv_data.personal_info.name}")
            print(f"   Overall Score: {result.overall_score:.1f}/100")
            print(f"   Job Fit: {result.fit_percentage:.1f}%")
            print(f"   Skills Found: {len(result.cv_data.skills)}")
            print(f"   Experience Entries: {len(result.cv_data.work_experience)}")
            print(f"   Education Entries: {len(result.cv_data.education)}")
            
            # Show top strengths and recommendations
            if result.strengths:
                print(f"\n💪 Top Strengths:")
                for i, strength in enumerate(result.strengths[:3], 1):
                    print(f"   {i}. {strength}")
            
            if result.recommendations:
                print(f"\n💡 Key Recommendations:")
                for i, rec in enumerate(result.recommendations[:2], 1):
                    print(f"   {i}. {rec}")
            
            # Export results
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
                results_file = f.name
            
            evaluator.export_results(results_file)
            print(f"\n📄 Results exported to: {Path(results_file).name}")
            
            # Clean up results file
            Path(results_file).unlink(missing_ok=True)
            
            return result
        else:
            print("❌ Evaluation failed")
            return None
    
    finally:
        # Clean up temporary files
        for temp_file in temp_files:
            try:
                Path(temp_file).unlink(missing_ok=True)
            except:
                pass


def main():
    """Run all advanced demos."""
    print("🚀 CV Evaluator Advanced Features Demo")
    print("Created by: Zied Boughdir (@zinzied)")
    print("=" * 60)
    
    try:
        # Demo 1: Custom criteria creation
        custom_criteria = demo_custom_criteria_creation()
        
        # Demo 2: Extract criteria from job description
        extracted_criteria = demo_criteria_from_job_description()
        
        # Demo 3: Participant evaluation with multiple files
        evaluation_result = demo_participant_evaluation()
        
        print("\n" + "=" * 60)
        print("🎉 All advanced demos completed successfully!")
        
        print("\n🚀 New Features Available:")
        print("1. Interactive Criteria Builder:")
        print("   python -m cv_evaluator create-criteria")
        
        print("\n2. Extract Criteria from Files:")
        print("   python -m cv_evaluator criteria-from-files job_description.txt")
        
        print("\n3. Participant Evaluation:")
        print("   python -m cv_evaluator evaluate-participant PART_001 \\")
        print("     --file cv.pdf:cv:\"Main CV\" \\")
        print("     --file cover_letter.txt:cover_letter:\"Cover letter\" \\")
        print("     --file portfolio.pdf:portfolio:\"Project portfolio\"")
        
        print("\n4. Web Interface with New Features:")
        print("   python run_web_app.py")
        print("   (Now includes Participant Evaluation and Criteria Creation)")
        
        print("\n💡 These features enable:")
        print("   • Custom evaluation criteria for any job role")
        print("   • Multi-file participant evaluation")
        print("   • Automatic criteria extraction from job descriptions")
        print("   • Enhanced candidate assessment with multiple documents")
        
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        return False
    
    return True


if __name__ == "__main__":
    main()
