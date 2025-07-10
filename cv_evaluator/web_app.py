"""
Streamlit web application for CV evaluation system.

Created by: Zied Boughdir (@zinzied)
GitHub: https://github.com/zinzied/cv-evaluation-system
"""

import streamlit as st
import tempfile
import json
import sys
from pathlib import Path
import pandas as pd
from typing import Optional

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import plotly.express as px
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    st.warning("⚠️ Plotly not installed. Charts will be limited.")

from cv_evaluator.core.evaluator import CVEvaluator
from cv_evaluator.core.batch_processor import BatchProcessor
from cv_evaluator.core.criteria_loader import criteria_manager
from cv_evaluator.core.models import EvaluationCriteria
from cv_evaluator.utils.exceptions import CVEvaluatorError


def main():
    """Main Streamlit application."""
    st.set_page_config(
        page_title="CV Evaluator",
        page_icon="📄",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("🤖 AI-Powered CV Evaluation System")
    st.markdown("Upload CV files and get detailed AI-powered analysis and scoring")
    st.markdown("*Created by [Zied Boughdir](https://github.com/zinzied)*")
    
    # Sidebar configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Job template selection
        templates = criteria_manager.list_job_templates()
        job_template = st.selectbox(
            "Job Template",
            options=["None"] + templates,
            help="Select a predefined job template for evaluation"
        )
        
        if job_template == "None":
            job_template = None
        
        # Evaluation mode
        mode = st.radio(
            "Evaluation Mode",
            ["Single CV", "Batch Processing"],
            help="Choose between single CV evaluation or batch processing"
        )
        
        # Report format
        report_format = st.selectbox(
            "Report Format",
            ["pdf", "html", "word"],
            help="Choose the output format for generated reports"
        )
    
    # Main content area
    if mode == "Single CV":
        single_cv_interface(job_template, report_format)
    else:
        batch_processing_interface(job_template, report_format)


def single_cv_interface(job_template: Optional[str], report_format: str):
    """Interface for single CV evaluation."""
    st.header("📄 Single CV Evaluation")
    
    # File upload
    uploaded_file = st.file_uploader(
        "Upload CV (PDF format)",
        type=['pdf'],
        help="Upload a PDF file containing the CV to evaluate"
    )
    
    if uploaded_file is not None:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_file_path = tmp_file.name
        
        # Display file info
        st.success(f"✅ Uploaded: {uploaded_file.name} ({uploaded_file.size} bytes)")
        
        # Evaluation button
        if st.button("🚀 Evaluate CV", type="primary"):
            try:
                with st.spinner("Analyzing CV... This may take a few moments."):
                    # Create evaluator
                    evaluator = CVEvaluator(job_template=job_template)
                    
                    # Evaluate CV
                    result = evaluator.evaluate_cv(tmp_file_path)
                    
                    # Display results
                    display_evaluation_results(result)
                    
                    # Generate and offer report download
                    with tempfile.NamedTemporaryFile(delete=False, suffix=f'.{report_format}') as report_file:
                        report_path = evaluator.generate_report(
                            result, report_file.name, format=report_format
                        )
                        
                        # Read report file for download
                        with open(report_path, 'rb') as f:
                            report_data = f.read()
                        
                        st.download_button(
                            label=f"📥 Download {report_format.upper()} Report",
                            data=report_data,
                            file_name=f"cv_evaluation_report.{report_format}",
                            mime=get_mime_type(report_format)
                        )
                        
            except CVEvaluatorError as e:
                st.error(f"❌ Evaluation failed: {e}")
            except Exception as e:
                st.error(f"❌ Unexpected error: {e}")
            finally:
                # Clean up temporary file
                Path(tmp_file_path).unlink(missing_ok=True)


def batch_processing_interface(job_template: Optional[str], report_format: str):
    """Interface for batch CV processing."""
    st.header("📁 Batch CV Processing")
    
    # Multiple file upload
    uploaded_files = st.file_uploader(
        "Upload multiple CVs (PDF format)",
        type=['pdf'],
        accept_multiple_files=True,
        help="Upload multiple PDF files for batch processing"
    )
    
    if uploaded_files:
        st.success(f"✅ Uploaded {len(uploaded_files)} files")
        
        # Display file list
        with st.expander("📋 Uploaded Files"):
            for i, file in enumerate(uploaded_files, 1):
                st.write(f"{i}. {file.name} ({file.size} bytes)")
        
        # Processing options
        col1, col2 = st.columns(2)
        with col1:
            generate_individual_reports = st.checkbox(
                "Generate individual reports",
                value=True,
                help="Generate a report for each CV"
            )
        
        with col2:
            max_workers = st.slider(
                "Concurrent workers",
                min_value=1,
                max_value=8,
                value=3,
                help="Number of CVs to process simultaneously"
            )
        
        # Process button
        if st.button("🚀 Process All CVs", type="primary"):
            try:
                with st.spinner(f"Processing {len(uploaded_files)} CVs..."):
                    # Save files temporarily
                    temp_files = []
                    for file in uploaded_files:
                        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                            tmp_file.write(file.getvalue())
                            temp_files.append(tmp_file.name)
                    
                    # Create batch processor
                    processor = BatchProcessor(
                        evaluation_criteria=criteria_manager.get_criteria("default", job_template),
                        max_workers=max_workers
                    )
                    
                    # Process files
                    with tempfile.TemporaryDirectory() as temp_dir:
                        results = processor.process_file_list(
                            file_paths=temp_files,
                            output_dir=temp_dir,
                            generate_reports=generate_individual_reports,
                            report_format=report_format
                        )
                        
                        # Display batch results
                        display_batch_results(results, uploaded_files)
                        
            except Exception as e:
                st.error(f"❌ Batch processing failed: {e}")
            finally:
                # Clean up temporary files
                for temp_file in temp_files:
                    Path(temp_file).unlink(missing_ok=True)


def display_evaluation_results(result):
    """Display evaluation results in the UI."""
    # Overall metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            "Overall Score",
            f"{result.overall_score:.1f}/100",
            help="Weighted overall evaluation score"
        )
    
    with col2:
        st.metric(
            "Job Fit",
            f"{result.fit_percentage:.1f}%",
            help="How well the candidate matches job requirements"
        )
    
    with col3:
        extraction_confidence = result.cv_data.extraction_confidence * 100
        st.metric(
            "Extraction Quality",
            f"{extraction_confidence:.1f}%",
            help="Quality of text extraction from PDF"
        )
    
    # Candidate information
    if result.cv_data.personal_info.name:
        st.subheader(f"👤 Candidate: {result.cv_data.personal_info.name}")
        
        info_cols = st.columns(4)
        with info_cols[0]:
            if result.cv_data.personal_info.email:
                st.write(f"📧 {result.cv_data.personal_info.email}")
        with info_cols[1]:
            if result.cv_data.personal_info.phone:
                st.write(f"📞 {result.cv_data.personal_info.phone}")
        with info_cols[2]:
            if result.cv_data.personal_info.linkedin:
                st.write(f"💼 LinkedIn")
        with info_cols[3]:
            if result.cv_data.personal_info.github:
                st.write(f"💻 GitHub")
    
    # Section scores visualization
    st.subheader("📊 Section Scores")
    
    # Create section scores chart
    section_data = []
    for score in result.section_scores:
        section_data.append({
            'Section': score.section.title(),
            'Score': score.score,
            'Max Score': score.max_score,
            'Percentage': (score.score / score.max_score) * 100
        })

    df_sections = pd.DataFrame(section_data)

    # Bar chart
    if PLOTLY_AVAILABLE:
        fig = px.bar(
            df_sections,
            x='Section',
            y='Percentage',
            title='Section Scores (%)',
            color='Percentage',
            color_continuous_scale='RdYlGn',
            range_color=[0, 100]
        )
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    else:
        # Fallback to simple bar chart
        st.bar_chart(df_sections.set_index('Section')['Percentage'])
    
    # Detailed section scores table
    with st.expander("📋 Detailed Section Scores"):
        for score in result.section_scores:
            col1, col2, col3 = st.columns([2, 1, 3])
            with col1:
                st.write(f"**{score.section.title()}**")
            with col2:
                percentage = (score.score / score.max_score) * 100
                st.write(f"{score.score:.1f}/{score.max_score:.1f} ({percentage:.1f}%)")
            with col3:
                st.write(score.feedback)
    
    # Skills analysis
    if result.cv_data.skills:
        st.subheader("🛠️ Skills Analysis")
        
        # Group skills by category
        skills_by_category = {}
        for skill in result.cv_data.skills:
            category = skill.category or 'General'
            if category not in skills_by_category:
                skills_by_category[category] = []
            skills_by_category[category].append(skill)
        
        # Display skills by category
        for category, skills in skills_by_category.items():
            with st.expander(f"{category.title()} Skills ({len(skills)})"):
                skill_cols = st.columns(3)
                for i, skill in enumerate(skills):
                    with skill_cols[i % 3]:
                        confidence_color = "🟢" if skill.confidence > 0.8 else "🟡" if skill.confidence > 0.5 else "🔴"
                        st.write(f"{confidence_color} {skill.name}")
    
    # Analysis insights
    col1, col2 = st.columns(2)
    
    with col1:
        if result.strengths:
            st.subheader("💪 Strengths")
            for strength in result.strengths:
                st.write(f"✅ {strength}")
    
    with col2:
        if result.weaknesses:
            st.subheader("🎯 Areas for Improvement")
            for weakness in result.weaknesses:
                st.write(f"⚠️ {weakness}")
    
    # Recommendations
    if result.recommendations:
        st.subheader("💡 Recommendations")
        for i, recommendation in enumerate(result.recommendations, 1):
            st.write(f"{i}. {recommendation}")


def display_batch_results(results, uploaded_files):
    """Display batch processing results."""
    st.subheader("📊 Batch Processing Results")
    
    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Files", results['total_files'])
    with col2:
        st.metric("Successful", results['successful'])
    with col3:
        st.metric("Failed", results['failed'])
    with col4:
        success_rate = (results['successful'] / results['total_files']) * 100
        st.metric("Success Rate", f"{success_rate:.1f}%")
    
    # Results table
    if results['results']:
        # Prepare data for table
        table_data = []
        for i, result in enumerate(results['results']):
            file_name = uploaded_files[i].name if i < len(uploaded_files) else f"File {i+1}"
            
            if result['success']:
                table_data.append({
                    'File': file_name,
                    'Status': '✅ Success',
                    'Candidate': result.get('candidate_name', 'Unknown'),
                    'Score': f"{result['overall_score']:.1f}",
                    'Fit %': f"{result['fit_percentage']:.1f}%",
                    'Processing Time': f"{result['processing_time']:.1f}s"
                })
            else:
                table_data.append({
                    'File': file_name,
                    'Status': '❌ Failed',
                    'Candidate': 'N/A',
                    'Score': 'N/A',
                    'Fit %': 'N/A',
                    'Processing Time': f"{result['processing_time']:.1f}s"
                })
        
        df_results = pd.DataFrame(table_data)
        st.dataframe(df_results, use_container_width=True)
        
        # Score distribution chart for successful results
        successful_results = [r for r in results['results'] if r['success']]
        if successful_results:
            scores = [r['overall_score'] for r in successful_results]

            if PLOTLY_AVAILABLE:
                fig = px.histogram(
                    x=scores,
                    nbins=10,
                    title='Score Distribution',
                    labels={'x': 'Overall Score', 'y': 'Number of Candidates'}
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                # Fallback to simple histogram
                st.subheader("Score Distribution")
                score_df = pd.DataFrame({'Scores': scores})
                st.bar_chart(score_df['Scores'].value_counts().sort_index())


def get_mime_type(format: str) -> str:
    """Get MIME type for file format."""
    mime_types = {
        'pdf': 'application/pdf',
        'html': 'text/html',
        'word': 'application/vnd.openxmlformats-officedocument.wordprocessingml.document'
    }
    return mime_types.get(format, 'application/octet-stream')


if __name__ == "__main__":
    main()
