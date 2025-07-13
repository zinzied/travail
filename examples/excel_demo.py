#!/usr/bin/env python3
"""
Excel integration demo for CV evaluation system.

Created by: Zied Boughdir (@zinzied)
GitHub: https://github.com/zinzied/cv-evaluation-system
"""

import sys
import tempfile
from pathlib import Path
import pandas as pd

# Add parent directory to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from cv_evaluator.excel.excel_processor import ExcelProcessor, ExcelBatchProcessor


def demo_excel_template():
    """Demo 1: Create Excel template for candidate import."""
    print("=== Demo 1: Excel Template Creation ===")
    
    try:
        processor = ExcelProcessor()
        
        # Create template
        template_file = "candidates_template_demo.xlsx"
        processor.create_excel_template(template_file)
        
        print(f"✅ Excel template created: {template_file}")
        
        # Show template structure
        df = pd.read_excel(template_file)
        print(f"   Columns: {list(df.columns)}")
        print(f"   Sample rows: {len(df)}")
        
        # Display sample data
        print("\n📋 Template Preview:")
        print(df.to_string(index=False))
        
        return template_file
        
    except Exception as e:
        print(f"❌ Template creation failed: {e}")
        return None


def demo_excel_evaluation():
    """Demo 2: Evaluate candidates from Excel file."""
    print("\n=== Demo 2: Excel Candidate Evaluation ===")
    
    try:
        # Create sample Excel file with candidates
        sample_data = {
            'candidate_id': ['CAND_001', 'CAND_002', 'CAND_003', 'CAND_004'],
            'name': ['Alice Johnson', 'Bob Smith', 'Carol Davis', 'David Wilson'],
            'email': ['alice.j@email.com', 'bob.s@email.com', 'carol.d@email.com', 'david.w@email.com'],
            'phone': ['+1-555-0101', '+1-555-0102', '+1-555-0103', '+1-555-0104'],
            'skills': [
                'Python, Machine Learning, SQL, TensorFlow, AWS',
                'Java, Spring Boot, React, Docker, Kubernetes',
                'JavaScript, Node.js, MongoDB, Vue.js, GraphQL',
                'C#, .NET Core, Azure, DevOps, Microservices'
            ],
            'experience': [
                'Senior Data Scientist at TechCorp (3 years); Data Analyst at StartupXYZ (2 years)',
                'Full Stack Developer at BigTech (4 years); Junior Developer at SmallCorp (1 year)',
                'Frontend Developer at WebAgency (3 years); Intern at DesignStudio (6 months)',
                'Backend Developer at Enterprise Inc (5 years); Software Engineer at CloudCorp (2 years)'
            ],
            'education': [
                'PhD Computer Science, MIT; MS Statistics, Stanford',
                'BS Computer Science, UC Berkeley; Coding Bootcamp, General Assembly',
                'BS Web Design, Art Institute; Frontend Certification, FreeCodeCamp',
                'MS Software Engineering, Carnegie Mellon; BS Computer Science, University of Washington'
            ],
            'cv_file_path': ['', '', '', ''],  # No CV files for this demo
            'status': ['pending', 'pending', 'pending', 'pending'],
            'notes': ['Strong ML background', 'Full-stack expertise', 'Creative frontend skills', 'Enterprise experience']
        }
        
        # Create Excel file
        candidates_file = "sample_candidates.xlsx"
        df_candidates = pd.DataFrame(sample_data)
        df_candidates.to_excel(candidates_file, index=False)
        
        print(f"✅ Sample candidates file created: {candidates_file}")
        print(f"   Candidates: {len(df_candidates)}")
        
        # Evaluate candidates
        processor = ExcelProcessor()
        print("\n🔄 Evaluating candidates from Excel...")
        
        results = processor.evaluate_candidates_from_excel(candidates_file, "default")
        
        if results:
            print(f"✅ Evaluation completed!")
            print(f"   Candidates evaluated: {len(results)}")
            
            # Show summary
            successful = len([r for r in results if r['overall_score'] > 0])
            avg_score = sum([r['overall_score'] for r in results if r['overall_score'] > 0]) / successful if successful > 0 else 0
            
            print(f"   Successful evaluations: {successful}")
            print(f"   Average score: {avg_score:.1f}/100")
            
            # Show top candidates
            print(f"\n🏆 Top Candidates:")
            sorted_results = sorted(results, key=lambda x: x['overall_score'], reverse=True)
            for i, candidate in enumerate(sorted_results[:3], 1):
                print(f"   {i}. {candidate['name']}: {candidate['overall_score']:.1f}/100 ({candidate['fit_percentage']:.1f}% fit)")
            
            # Export results
            results_file = "evaluation_results_demo.xlsx"
            processor.export_results_to_excel(results, results_file)
            print(f"\n📊 Results exported to: {results_file}")
            
            # Show Excel sheets created
            excel_file = pd.ExcelFile(results_file)
            print(f"   Excel sheets: {excel_file.sheet_names}")
            
            return results_file
        
        else:
            print("❌ No candidates were evaluated")
            return None
            
    except Exception as e:
        print(f"❌ Excel evaluation failed: {e}")
        return None


def demo_batch_processing():
    """Demo 3: Batch process CVs and export to Excel."""
    print("\n=== Demo 3: Batch CV Processing to Excel ===")
    
    try:
        # Create sample CV files
        sample_cvs = {
            "john_doe_cv.txt": """
John Doe
john.doe@email.com
+1-555-0123

PROFESSIONAL SUMMARY
Experienced software engineer with 5+ years in full-stack development.

TECHNICAL SKILLS
Programming: Python, JavaScript, Java
Web: React, Node.js, Django
Cloud: AWS, Docker
Databases: PostgreSQL, MongoDB

EXPERIENCE
Senior Software Engineer | TechCorp Inc. | 2021 - Present
• Lead development of microservices architecture
• Mentor team of 3 junior developers

Software Engineer | StartupXYZ | 2019 - 2021
• Developed RESTful APIs and web applications

EDUCATION
MS Computer Science | Stanford University | 2019
BS Software Engineering | UC Berkeley | 2017
""",
            
            "jane_smith_cv.txt": """
Jane Smith
jane.smith@email.com
+1-555-0124

PROFESSIONAL SUMMARY
Data scientist with expertise in machine learning and analytics.

TECHNICAL SKILLS
Programming: Python, R, SQL
ML/AI: TensorFlow, PyTorch, scikit-learn
Data: Pandas, NumPy, Matplotlib
Cloud: AWS, Azure

EXPERIENCE
Senior Data Scientist | DataCorp | 2020 - Present
• Built ML models for customer segmentation
• Improved prediction accuracy by 25%

Data Analyst | Analytics Inc. | 2018 - 2020
• Created dashboards and reports

EDUCATION
PhD Statistics | MIT | 2018
MS Data Science | UC Berkeley | 2016
""",
            
            "mike_johnson_cv.txt": """
Mike Johnson
mike.johnson@email.com
+1-555-0125

PROFESSIONAL SUMMARY
DevOps engineer specializing in cloud infrastructure and automation.

TECHNICAL SKILLS
Cloud: AWS, Azure, GCP
DevOps: Docker, Kubernetes, Jenkins
Programming: Python, Bash, Go
Infrastructure: Terraform, Ansible

EXPERIENCE
DevOps Engineer | CloudTech | 2019 - Present
• Managed cloud infrastructure for 100+ services
• Reduced deployment time by 60%

System Administrator | Enterprise Corp | 2017 - 2019
• Maintained Linux servers and networks

EDUCATION
BS Computer Science | University of Texas | 2017
AWS Certified Solutions Architect | 2020
"""
        }
        
        # Create temporary CV files
        temp_files = []
        for filename, content in sample_cvs.items():
            temp_file = Path(filename)
            temp_file.write_text(content)
            temp_files.append(temp_file)
        
        print(f"✅ Created {len(temp_files)} sample CV files")
        
        # Create temporary folder
        cv_folder = Path("temp_cvs")
        cv_folder.mkdir(exist_ok=True)
        
        # Move files to folder
        for temp_file in temp_files:
            temp_file.rename(cv_folder / temp_file.name)
        
        # Process folder
        batch_processor = ExcelBatchProcessor()
        print(f"\n🔄 Processing CV folder: {cv_folder}")
        
        results_file = "batch_results_demo.xlsx"
        output_file = batch_processor.process_cv_folder_to_excel(str(cv_folder), results_file)
        
        print(f"✅ Batch processing completed!")
        print(f"   Results file: {output_file}")
        
        # Show results summary
        df_results = pd.read_excel(output_file, sheet_name='Evaluation Results')
        print(f"   CVs processed: {len(df_results)}")
        print(f"   Average score: {df_results['overall_score'].mean():.1f}/100")
        
        # Show Excel structure
        excel_file = pd.ExcelFile(output_file)
        print(f"   Excel sheets: {excel_file.sheet_names}")
        
        # Clean up
        import shutil
        shutil.rmtree(cv_folder)
        
        return output_file
        
    except Exception as e:
        print(f"❌ Batch processing failed: {e}")
        return None


def demo_excel_analysis():
    """Demo 4: Analyze results in Excel format."""
    print("\n=== Demo 4: Excel Results Analysis ===")
    
    try:
        # Use results from previous demo
        results_file = "evaluation_results_demo.xlsx"
        
        if not Path(results_file).exists():
            print(f"⚠️ Results file not found: {results_file}")
            return
        
        # Read and analyze results
        excel_file = pd.ExcelFile(results_file)
        
        print(f"📊 Analyzing Excel file: {results_file}")
        print(f"   Available sheets: {excel_file.sheet_names}")
        
        # Main results
        df_results = pd.read_excel(results_file, sheet_name='Evaluation Results')
        print(f"\n📋 Main Results:")
        print(f"   Total candidates: {len(df_results)}")
        print(f"   Average overall score: {df_results['overall_score'].mean():.1f}")
        print(f"   Score range: {df_results['overall_score'].min():.1f} - {df_results['overall_score'].max():.1f}")
        
        # Summary statistics
        df_summary = pd.read_excel(results_file, sheet_name='Summary Statistics')
        print(f"\n📈 Summary Statistics:")
        for _, row in df_summary.iterrows():
            print(f"   {row['Metric']}: {row['Value']}")
        
        # Top candidates
        df_top = pd.read_excel(results_file, sheet_name='Top Candidates')
        print(f"\n🏆 Top Candidates:")
        for i, (_, candidate) in enumerate(df_top.head(3).iterrows(), 1):
            print(f"   {i}. {candidate['name']}: {candidate['overall_score']:.1f}/100")
        
        # Skills analysis
        df_skills = pd.read_excel(results_file, sheet_name='Skills Analysis')
        print(f"\n🎯 Score Distribution:")
        for _, row in df_skills.iterrows():
            print(f"   {row['Score Range']}: {row['Number of Candidates']} candidates ({row['Percentage']})")
        
        return True
        
    except Exception as e:
        print(f"❌ Excel analysis failed: {e}")
        return False


def main():
    """Run all Excel integration demos."""
    print("🚀 Excel Integration Demo for CV Evaluator")
    print("Created by: Zied Boughdir (@zinzied)")
    print("=" * 60)
    
    try:
        # Demo 1: Create template
        template_file = demo_excel_template()
        
        # Demo 2: Evaluate from Excel
        results_file = demo_excel_evaluation()
        
        # Demo 3: Batch processing
        batch_file = demo_batch_processing()
        
        # Demo 4: Analyze results
        demo_excel_analysis()
        
        print("\n" + "=" * 60)
        print("🎉 Excel Integration Demo Completed!")
        
        print("\n🚀 Excel Features Available:")
        print("1. Create Excel Template:")
        print("   python -m cv_evaluator excel-template")
        
        print("\n2. Evaluate from Excel:")
        print("   python -m cv_evaluator excel-evaluate candidates.xlsx")
        
        print("\n3. Batch Process to Excel:")
        print("   python -m cv_evaluator excel-batch cv_folder/")
        
        print("\n4. Web Interface:")
        print("   python run_web_app.py")
        print("   (Select 'Excel Integration' mode)")
        
        print("\n💡 Excel Integration Benefits:")
        print("   • Import candidate data from spreadsheets")
        print("   • Bulk evaluate multiple candidates")
        print("   • Export detailed results with multiple sheets")
        print("   • Statistical analysis and summaries")
        print("   • Easy data manipulation and filtering")
        print("   • Professional reporting format")
        
        print("\n📊 Excel Output Includes:")
        print("   • Evaluation Results - Detailed candidate scores")
        print("   • Summary Statistics - Overall metrics")
        print("   • Top Candidates - Best performers")
        print("   • Skills Analysis - Score distributions")
        
        # Clean up demo files
        cleanup_files = [
            "candidates_template_demo.xlsx",
            "sample_candidates.xlsx", 
            "evaluation_results_demo.xlsx",
            "batch_results_demo.xlsx"
        ]
        
        for file in cleanup_files:
            try:
                Path(file).unlink(missing_ok=True)
            except:
                pass
        
        return True
        
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
