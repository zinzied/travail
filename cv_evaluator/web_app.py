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
from cv_evaluator.core.interactive_criteria import InteractiveCriteriaBuilder, CriteriaFromFiles
from cv_evaluator.core.participant_evaluator import ParticipantEvaluator
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
            ["Single CV", "Batch Processing", "Participant Evaluation", "Create Criteria"],
            help="Choose evaluation mode"
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
    elif mode == "Batch Processing":
        batch_processing_interface(job_template, report_format)
    elif mode == "Participant Evaluation":
        participant_evaluation_interface(job_template, report_format)
    elif mode == "Create Criteria":
        criteria_creation_interface()


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


def participant_evaluation_interface(job_template: Optional[str], report_format: str):
    """Interface for participant evaluation with multiple files."""
    st.header("👥 Participant Evaluation")
    st.markdown("Evaluate participants using multiple files (CV, cover letter, portfolio, etc.)")

    # Participant information
    col1, col2 = st.columns(2)
    with col1:
        participant_id = st.text_input("Participant ID", placeholder="e.g., PART_001")
    with col2:
        participant_name = st.text_input("Participant Name (optional)", placeholder="e.g., John Doe")

    if not participant_id:
        st.warning("Please enter a Participant ID to continue.")
        return

    # File upload section
    st.subheader("📁 Upload Participant Files")

    # File type options
    file_types = {
        "CV/Resume": "cv",
        "Cover Letter": "cover_letter",
        "Portfolio": "portfolio",
        "Transcript": "transcript",
        "Other": "other"
    }

    uploaded_files = []

    # Multiple file uploaders
    for display_name, file_type in file_types.items():
        with st.expander(f"Upload {display_name}"):
            files = st.file_uploader(
                f"Choose {display_name} files",
                type=['pdf', 'txt'],
                accept_multiple_files=True,
                key=f"files_{file_type}"
            )

            if files:
                for file in files:
                    description = st.text_input(
                        f"Description for {file.name}",
                        key=f"desc_{file_type}_{file.name}",
                        placeholder="Optional description"
                    )
                    uploaded_files.append({
                        'file': file,
                        'type': file_type,
                        'description': description
                    })

    if uploaded_files:
        st.success(f"✅ {len(uploaded_files)} files ready for processing")

        # Display file summary
        with st.expander("📋 File Summary"):
            for i, file_info in enumerate(uploaded_files, 1):
                st.write(f"{i}. **{file_info['file'].name}** ({file_info['type']}) - {file_info['description'] or 'No description'}")

        # Evaluation button
        if st.button("🚀 Evaluate Participant", type="primary"):
            try:
                with st.spinner("Processing participant files and evaluating..."):
                    # Load criteria
                    evaluation_criteria = criteria_manager.get_criteria("default", job_template)

                    # Create participant evaluator
                    evaluator = ParticipantEvaluator(evaluation_criteria)

                    # Save files temporarily and add to evaluator
                    participant_files = []
                    temp_files = []

                    for file_info in uploaded_files:
                        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file_info['file'].name.split('.')[-1]}") as tmp_file:
                            tmp_file.write(file_info['file'].getvalue())
                            temp_file_path = tmp_file.name
                            temp_files.append(temp_file_path)

                            participant_files.append({
                                'path': temp_file_path,
                                'type': file_info['type'],
                                'description': file_info['description']
                            })

                    # Add participant
                    evaluator.add_participant_files(participant_id, participant_files)
                    if participant_name:
                        evaluator.participants[participant_id].name = participant_name

                    # Process and evaluate
                    evaluator.process_participant_files(participant_id)
                    result = evaluator.evaluate_participant(participant_id)

                    if result:
                        # Display results
                        display_participant_evaluation_results(participant_id, result, evaluator)

                        # Generate and offer report download
                        with tempfile.NamedTemporaryFile(delete=False, suffix=f'.{report_format}') as report_file:
                            report_path = evaluator.evaluator.generate_report(
                                result, report_file.name, format=report_format
                            )

                            # Read report file for download
                            with open(report_path, 'rb') as f:
                                report_data = f.read()

                            st.download_button(
                                label=f"📥 Download {report_format.upper()} Report",
                                data=report_data,
                                file_name=f"{participant_id}_evaluation_report.{report_format}",
                                mime=get_mime_type(report_format)
                            )
                    else:
                        st.error("❌ Failed to evaluate participant")

                    # Clean up temporary files
                    for temp_file in temp_files:
                        try:
                            Path(temp_file).unlink(missing_ok=True)
                        except:
                            pass

            except Exception as e:
                st.error(f"❌ Evaluation failed: {e}")


def criteria_creation_interface():
    """Interface for creating custom evaluation criteria."""
    st.header("⚙️ Create Custom Evaluation Criteria")
    st.markdown("Define your own evaluation criteria or extract them from job description files.")

    # Choose creation method
    creation_method = st.radio(
        "How would you like to create criteria?",
        ["Interactive Builder", "Extract from Files", "Upload Criteria File"],
        help="Choose your preferred method for creating evaluation criteria"
    )

    if creation_method == "Interactive Builder":
        st.subheader("🔧 Interactive Criteria Builder")
        st.markdown("*Note: For full interactive experience, use the command line: `python -m cv_evaluator create-criteria`*")

        # Simplified web version
        with st.form("criteria_form"):
            st.write("**Basic Information**")
            job_title = st.text_input("Job Title", placeholder="e.g., Senior Software Engineer")
            department = st.text_input("Department", placeholder="e.g., Engineering")

            st.write("**Required Skills** (one per line)")
            required_skills_text = st.text_area("Required Skills", placeholder="Python\nSQL\nCommunication")

            st.write("**Preferred Skills** (one per line)")
            preferred_skills_text = st.text_area("Preferred Skills", placeholder="Machine Learning\nAWS\nDocker")

            st.write("**Experience Requirements**")
            min_years = st.number_input("Minimum Years of Experience", min_value=0, value=2)

            st.write("**Scoring Weights** (must sum to 1.0)")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                skills_weight = st.number_input("Skills", min_value=0.0, max_value=1.0, value=0.4, step=0.05)
            with col2:
                experience_weight = st.number_input("Experience", min_value=0.0, max_value=1.0, value=0.3, step=0.05)
            with col3:
                education_weight = st.number_input("Education", min_value=0.0, max_value=1.0, value=0.2, step=0.05)
            with col4:
                additional_weight = st.number_input("Additional", min_value=0.0, max_value=1.0, value=0.1, step=0.05)

            filename = st.text_input("Criteria Filename", value="custom_criteria", help="Name for saving the criteria")

            submitted = st.form_submit_button("Create Criteria")

            if submitted:
                # Validate weights
                total_weight = skills_weight + experience_weight + education_weight + additional_weight
                if abs(total_weight - 1.0) > 0.01:
                    st.error(f"Scoring weights must sum to 1.0 (current sum: {total_weight:.2f})")
                    return

                try:
                    # Parse skills
                    required_skills = [s.strip() for s in required_skills_text.split('\n') if s.strip()]
                    preferred_skills = [s.strip() for s in preferred_skills_text.split('\n') if s.strip()]

                    # Create criteria
                    criteria_data = {
                        'required_skills': required_skills,
                        'preferred_skills': preferred_skills,
                        'min_experience_years': min_years,
                        'scoring_weights': {
                            'skills': skills_weight,
                            'experience': experience_weight,
                            'education': education_weight,
                            'additional': additional_weight
                        },
                        'max_score': 100
                    }

                    criteria = EvaluationCriteria(**criteria_data)

                    # Save criteria
                    builder = InteractiveCriteriaBuilder()
                    filepath = builder.save_criteria_to_file(criteria, filename)

                    st.success(f"✅ Criteria created successfully!")
                    st.info(f"Saved as: {filename}.yaml")
                    st.info(f"Use with: --criteria {filename}")

                except Exception as e:
                    st.error(f"❌ Failed to create criteria: {e}")

    elif creation_method == "Extract from Files":
        st.subheader("📄 Extract from Job Description Files")

        uploaded_files = st.file_uploader(
            "Upload job description or requirement files",
            type=['txt', 'json'],
            accept_multiple_files=True,
            help="Upload text files containing job descriptions or requirements"
        )

        if uploaded_files:
            filename = st.text_input("Output Criteria Filename", value="extracted_criteria")

            if st.button("Extract Criteria"):
                try:
                    # Save uploaded files temporarily
                    temp_files = []
                    for file in uploaded_files:
                        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file.name.split('.')[-1]}") as tmp_file:
                            tmp_file.write(file.getvalue())
                            temp_files.append(tmp_file.name)

                    # Extract criteria
                    extractor = CriteriaFromFiles()
                    criteria = extractor.extract_criteria_from_files(temp_files)

                    if criteria:
                        # Save criteria
                        builder = InteractiveCriteriaBuilder()
                        filepath = builder.save_criteria_to_file(criteria, filename)

                        st.success(f"✅ Criteria extracted successfully!")
                        st.info(f"Saved as: {filename}.yaml")

                        # Display extracted criteria
                        st.subheader("Extracted Criteria Preview")
                        col1, col2 = st.columns(2)
                        with col1:
                            st.write("**Required Skills:**")
                            for skill in criteria.required_skills:
                                st.write(f"• {skill}")
                        with col2:
                            st.write("**Preferred Skills:**")
                            for skill in criteria.preferred_skills:
                                st.write(f"• {skill}")

                        st.write(f"**Minimum Experience:** {criteria.min_experience_years} years")

                    # Clean up
                    for temp_file in temp_files:
                        try:
                            Path(temp_file).unlink(missing_ok=True)
                        except:
                            pass

                except Exception as e:
                    st.error(f"❌ Failed to extract criteria: {e}")


def display_participant_evaluation_results(participant_id: str, result, evaluator):
    """Display participant evaluation results in the web interface."""
    participant = evaluator.participants[participant_id]

    # Participant summary
    st.subheader(f"📊 Evaluation Results for {participant.name or participant_id}")

    # Overall metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Overall Score", f"{result.overall_score:.1f}/100")
    with col2:
        st.metric("Job Fit", f"{result.fit_percentage:.1f}%")
    with col3:
        st.metric("Files Processed", len(participant.files))
    with col4:
        successful_files = len([f for f in participant.files if f.processing_status == "completed"])
        st.metric("Success Rate", f"{(successful_files/len(participant.files)*100):.0f}%")

    # Files processing status
    st.subheader("📁 File Processing Status")
    files_data = []
    for file_obj in participant.files:
        files_data.append({
            'File': file_obj.file_path.name,
            'Type': file_obj.file_type.title(),
            'Status': file_obj.processing_status.title(),
            'Description': file_obj.description or 'N/A'
        })

    df_files = pd.DataFrame(files_data)
    st.dataframe(df_files, use_container_width=True)

    # Display standard evaluation results
    display_evaluation_results(result)


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
