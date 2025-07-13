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
from cv_evaluator.excel.excel_processor import ExcelProcessor, ExcelBatchProcessor
from cv_evaluator.ai.free_models import get_ai_response, list_available_models, auto_select_ai_model
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
            ["Single CV", "Batch Processing", "Participant Evaluation", "Create Criteria", "AI Chat", "Excel Integration"],
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
    elif mode == "AI Chat":
        ai_chat_interface()
    elif mode == "Excel Integration":
        excel_integration_interface(job_template)


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
        import tempfile as tmp_module
        with tmp_module.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
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
                    with tmp_module.NamedTemporaryFile(delete=False, suffix=f'.{report_format}') as report_file:
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
                        with tmp_module.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
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
                        with tmp_module.NamedTemporaryFile(delete=False, suffix=f".{file_info['file'].name.split('.')[-1]}") as tmp_file:
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
                        with tmp_module.NamedTemporaryFile(delete=False, suffix=f'.{report_format}') as report_file:
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
                        with tmp_module.NamedTemporaryFile(delete=False, suffix=f".{file.name.split('.')[-1]}") as tmp_file:
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


def ai_chat_interface():
    """AI chat interface for CV evaluation."""
    st.header("💬 AI Chat Assistant")
    st.markdown("Upload a CV and chat with AI about the candidate's qualifications")

    # Initialize session state for chat
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'current_cv_data' not in st.session_state:
        st.session_state.current_cv_data = None
    if 'evaluator' not in st.session_state:
        st.session_state.evaluator = None

    # Model selection
    with st.expander("🤖 AI Model Settings"):
        available_models = list_available_models()

        if not any(available_models.values()):
            st.warning("⚠️ No AI models are currently available")
            st.markdown("""
            **To enable AI chat features, set up one of these free options:**

            1. **Ollama** (Recommended):
               - Download: https://ollama.ai
               - Install a model: `ollama pull llama2`
               - Start server: `ollama serve`

            2. **Hugging Face Transformers**:
               - Install: `pip install transformers torch`
               - Models download automatically

            3. **LocalAI or compatible API**:
               - Set up at http://localhost:8080
            """)
        else:
            # Show available models
            st.write("**Available Models:**")
            for model_name, is_available in available_models.items():
                status = "✅" if is_available else "❌"
                st.write(f"{status} {model_name}")

            # Auto-select best model
            if st.button("🔄 Auto-select Best Model"):
                selected = auto_select_ai_model()
                if selected:
                    st.success(f"Selected model: {selected}")
                    st.rerun()
                else:
                    st.error("No models available")

    # CV upload section
    st.subheader("📄 Upload CV")

    uploaded_file = st.file_uploader(
        "Choose a CV file",
        type=['pdf', 'txt'],
        help="Upload a PDF or text file containing the CV"
    )

    if uploaded_file is not None:
        # Process the uploaded CV
        with st.spinner("Processing CV..."):
            success = process_uploaded_cv_for_chat(uploaded_file)

            if success:
                candidate_name = st.session_state.current_cv_data.personal_info.name or 'Unknown candidate'
                st.success(f"✅ CV processed: {candidate_name}")

                # Show CV summary
                with st.expander("📋 CV Summary"):
                    display_cv_summary_for_chat()
            else:
                st.error("❌ Failed to process CV")

    # Chat section
    if st.session_state.current_cv_data:
        st.subheader("💬 Chat about this CV")

        # Display chat history
        if st.session_state.chat_history:
            st.write("**Chat History:**")
            for i, (question, answer) in enumerate(st.session_state.chat_history):
                with st.container():
                    st.write(f"**You:** {question}")
                    st.write(f"**Assistant:** {answer}")
                    st.divider()

        # Quick question buttons
        st.write("**Quick Questions:**")
        col1, col2, col3 = st.columns(3)

        with col1:
            if st.button("🎯 Overall Assessment"):
                ask_question_about_cv("What is your overall assessment of this candidate?")

        with col2:
            if st.button("💪 Key Strengths"):
                ask_question_about_cv("What are the key strengths of this candidate?")

        with col3:
            if st.button("📈 Areas for Improvement"):
                ask_question_about_cv("What areas should this candidate improve?")

        # Custom question input
        user_question = st.text_input(
            "Ask a question about this CV:",
            placeholder="e.g., Does this candidate have experience with Python?",
            key="chat_input"
        )

        col1, col2 = st.columns([1, 4])
        with col1:
            if st.button("Send", type="primary"):
                if user_question.strip():
                    ask_question_about_cv(user_question)
                    st.rerun()

        with col2:
            if st.button("🗑️ Clear Chat"):
                st.session_state.chat_history = []
                st.rerun()
    else:
        st.info("👆 Please upload a CV to start chatting about the candidate")


def process_uploaded_cv_for_chat(uploaded_file) -> bool:
    """Process uploaded CV file for chat interface."""
    try:
        # Save file temporarily
        import tempfile as tmp_module
        with tmp_module.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_file_path = tmp_file.name

        # Create evaluator and process CV
        st.session_state.evaluator = CVEvaluator()

        if uploaded_file.name.endswith('.pdf'):
            # Extract from PDF
            extraction_result = st.session_state.evaluator.pdf_extractor.extract_text(tmp_file_path)
            cv_text = extraction_result.get('text', '')
        else:
            # Read text file
            cv_text = Path(tmp_file_path).read_text(encoding='utf-8')

        # Parse CV
        st.session_state.current_cv_data = st.session_state.evaluator.cv_parser.parse_cv(cv_text)

        # Clear chat history when new CV is uploaded
        st.session_state.chat_history = []

        # Clean up
        Path(tmp_file_path).unlink(missing_ok=True)

        return True

    except Exception as e:
        st.error(f"Error processing CV: {e}")
        return False


def display_cv_summary_for_chat():
    """Display a summary of the processed CV for chat."""
    cv_data = st.session_state.current_cv_data
    if not cv_data:
        return

    col1, col2 = st.columns(2)

    with col1:
        st.write("**Personal Information:**")
        if cv_data.personal_info.name:
            st.write(f"Name: {cv_data.personal_info.name}")
        if cv_data.personal_info.email:
            st.write(f"Email: {cv_data.personal_info.email}")

        st.write(f"**Skills Found:** {len(cv_data.skills)}")
        if cv_data.skills:
            skills_preview = ", ".join([skill.name for skill in cv_data.skills[:5]])
            st.write(f"Top skills: {skills_preview}...")

    with col2:
        st.write(f"**Work Experience:** {len(cv_data.work_experience)} entries")
        if cv_data.work_experience:
            recent_job = cv_data.work_experience[0]
            st.write(f"Recent: {recent_job.position} at {recent_job.company}")

        st.write(f"**Education:** {len(cv_data.education)} entries")
        if cv_data.education:
            recent_edu = cv_data.education[0]
            st.write(f"Latest: {recent_edu.degree} from {recent_edu.institution}")


def ask_question_about_cv(question: str):
    """Process a user question about the CV."""
    cv_data = st.session_state.current_cv_data
    if not cv_data:
        return

    try:
        # Get CV context
        cv_context = get_cv_context_for_chat(cv_data)

        # Create prompt for AI
        prompt = f"""
You are a CV evaluation expert. Here's information about a candidate:

{cv_context}

User question: {question}

Please provide a helpful, professional response about this candidate based on the CV information.
Keep your response concise and focused on the question asked.
"""

        # Get AI response
        answer = get_ai_response(prompt, max_tokens=500)

        # Add to chat history
        st.session_state.chat_history.append((question, answer))

    except Exception as e:
        error_msg = f"Sorry, I encountered an error: {e}"
        st.session_state.chat_history.append((question, error_msg))


def get_cv_context_for_chat(cv_data) -> str:
    """Get CV context for AI prompts."""
    context_parts = []

    # Personal info
    if cv_data.personal_info.name:
        context_parts.append(f"Candidate: {cv_data.personal_info.name}")

    # Skills
    if cv_data.skills:
        skills = ", ".join([skill.name for skill in cv_data.skills[:10]])
        context_parts.append(f"Skills: {skills}")

    # Experience
    if cv_data.work_experience:
        exp_summary = []
        for exp in cv_data.work_experience[:3]:
            exp_summary.append(f"{exp.position} at {exp.company}")
        context_parts.append(f"Experience: {'; '.join(exp_summary)}")

    # Education
    if cv_data.education:
        edu_summary = []
        for edu in cv_data.education[:2]:
            edu_summary.append(f"{edu.degree} from {edu.institution}")
        context_parts.append(f"Education: {'; '.join(edu_summary)}")

    return "\n".join(context_parts)


def excel_integration_interface(job_template: Optional[str]):
    """Excel integration interface for importing/exporting candidate data."""
    st.header("📊 Excel Integration")
    st.markdown("Import candidates from Excel, evaluate them, and export results back to Excel")

    # Choose Excel operation
    excel_operation = st.radio(
        "Choose Excel Operation",
        ["Create Template", "Import & Evaluate", "Batch Process Folder"],
        help="Select what you want to do with Excel"
    )

    if excel_operation == "Create Template":
        st.subheader("📋 Create Excel Template")
        st.markdown("Download a template Excel file for importing candidate data")

        if st.button("📥 Download Excel Template", type="primary"):
            try:
                processor = ExcelProcessor()

                # Create template in memory
                import tempfile as tmp_module
                with tmp_module.NamedTemporaryFile(suffix='.xlsx', delete=False) as tmp_file:
                    template_path = processor.create_excel_template(tmp_file.name)

                    # Read file for download
                    with open(template_path, 'rb') as f:
                        template_data = f.read()

                    st.download_button(
                        label="📥 Download Template",
                        data=template_data,
                        file_name="candidates_template.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )

                    # Clean up
                    Path(template_path).unlink(missing_ok=True)

                st.success("✅ Template ready for download!")

                # Show template structure
                with st.expander("📋 Template Structure"):
                    st.write("**Required Columns:**")
                    st.write("• `candidate_id` - Unique identifier for each candidate")
                    st.write("• `name` - Candidate's full name")
                    st.write("• `email` - Email address")
                    st.write("• `phone` - Phone number")
                    st.write("• `skills` - Comma-separated list of skills")
                    st.write("• `experience` - Work experience (semicolon-separated entries)")
                    st.write("• `education` - Education background")
                    st.write("• `cv_file_path` - Path to CV file (optional)")
                    st.write("• `status` - Current status (pending, reviewed, etc.)")
                    st.write("• `notes` - Additional notes")

            except Exception as e:
                st.error(f"❌ Failed to create template: {e}")

    elif excel_operation == "Import & Evaluate":
        st.subheader("📤 Import & Evaluate Candidates")
        st.markdown("Upload an Excel file with candidate data and get evaluation results")

        # File upload
        uploaded_excel = st.file_uploader(
            "Upload Excel file with candidates",
            type=['xlsx', 'xls'],
            help="Upload Excel file containing candidate information"
        )

        if uploaded_excel is not None:
            try:
                # Save uploaded file temporarily
                import tempfile as tmp_module
                with tmp_module.NamedTemporaryFile(suffix='.xlsx', delete=False) as tmp_file:
                    tmp_file.write(uploaded_excel.getvalue())
                    excel_path = tmp_file.name

                # Preview data
                import pandas as pd
                df_preview = pd.read_excel(excel_path)

                st.write("**Data Preview:**")
                st.dataframe(df_preview.head(), use_container_width=True)

                st.write(f"**Total candidates:** {len(df_preview)}")

                # Evaluation settings
                col1, col2 = st.columns(2)
                with col1:
                    criteria_name = st.selectbox(
                        "Evaluation Criteria",
                        ["default", "software_engineer", "data_scientist", "product_manager"],
                        help="Choose evaluation criteria"
                    )

                with col2:
                    include_ai = st.checkbox("Use AI Enhancement", value=True, help="Enable AI-powered insights")

                # Evaluate button
                if st.button("🚀 Evaluate All Candidates", type="primary"):
                    try:
                        with st.spinner("Evaluating candidates... This may take a few minutes."):
                            processor = ExcelProcessor()
                            results = processor.evaluate_candidates_from_excel(excel_path, criteria_name)

                        if results:
                            st.success(f"✅ Evaluated {len(results)} candidates!")

                            # Show summary
                            successful = len([r for r in results if r['overall_score'] > 0])
                            failed = len(results) - successful
                            avg_score = sum([r['overall_score'] for r in results if r['overall_score'] > 0]) / successful if successful > 0 else 0

                            col1, col2, col3, col4 = st.columns(4)
                            with col1:
                                st.metric("Total", len(results))
                            with col2:
                                st.metric("Successful", successful)
                            with col3:
                                st.metric("Failed", failed)
                            with col4:
                                st.metric("Avg Score", f"{avg_score:.1f}")

                            # Show results table
                            st.subheader("📊 Evaluation Results")
                            results_df = pd.DataFrame(results)
                            st.dataframe(results_df, use_container_width=True)

                            # Export results
                            with tmp_module.NamedTemporaryFile(suffix='.xlsx', delete=False) as tmp_result:
                                output_path = processor.export_results_to_excel(results, tmp_result.name)

                                # Read for download
                                with open(output_path, 'rb') as f:
                                    results_data = f.read()

                                st.download_button(
                                    label="📥 Download Results Excel",
                                    data=results_data,
                                    file_name=f"evaluation_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                                )

                                # Clean up
                                Path(output_path).unlink(missing_ok=True)

                        else:
                            st.error("❌ No candidates were successfully evaluated")

                    except Exception as e:
                        st.error(f"❌ Evaluation failed: {e}")

                # Clean up
                Path(excel_path).unlink(missing_ok=True)

            except Exception as e:
                st.error(f"❌ Failed to process Excel file: {e}")

    elif excel_operation == "Batch Process Folder":
        st.subheader("📁 Batch Process CV Folder")
        st.markdown("Process all CVs in a folder and export results to Excel")

        # Folder selection (simulated with file upload)
        st.info("💡 Upload multiple CV files to simulate folder processing")

        uploaded_cvs = st.file_uploader(
            "Upload CV files",
            type=['pdf', 'txt'],
            accept_multiple_files=True,
            help="Upload multiple CV files for batch processing"
        )

        if uploaded_cvs:
            st.write(f"**Files uploaded:** {len(uploaded_cvs)}")

            # Show file list
            with st.expander("📋 Uploaded Files"):
                for i, file in enumerate(uploaded_cvs, 1):
                    st.write(f"{i}. {file.name}")

            # Evaluation settings
            criteria_name = st.selectbox(
                "Evaluation Criteria",
                ["default", "software_engineer", "data_scientist", "product_manager"],
                help="Choose evaluation criteria",
                key="batch_criteria"
            )

            if st.button("🚀 Process All CVs", type="primary"):
                try:
                    with st.spinner("Processing CVs and generating Excel report..."):
                        # Save files temporarily
                        temp_files = []
                        results = []

                        evaluator = CVEvaluator(criteria_name=criteria_name)

                        for i, uploaded_file in enumerate(uploaded_cvs, 1):
                            try:
                                # Save file temporarily
                                with tmp_module.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp_file:
                                    tmp_file.write(uploaded_file.getvalue())
                                    temp_file_path = tmp_file.name
                                    temp_files.append(temp_file_path)

                                # Evaluate CV
                                result = evaluator.evaluate_cv(temp_file_path)

                                if result:
                                    candidate_result = {
                                        'candidate_id': f"CV_{i:03d}",
                                        'name': result.cv_data.personal_info.name or uploaded_file.name.split('.')[0],
                                        'email': result.cv_data.personal_info.email or '',
                                        'phone': result.cv_data.personal_info.phone or '',
                                        'cv_file': uploaded_file.name,
                                        'overall_score': result.overall_score,
                                        'fit_percentage': result.fit_percentage,
                                        'skills_score': next((s.score for s in result.section_scores if s.section == 'skills'), 0),
                                        'experience_score': next((s.score for s in result.section_scores if s.section == 'experience'), 0),
                                        'education_score': next((s.score for s in result.section_scores if s.section == 'education'), 0),
                                        'additional_score': next((s.score for s in result.section_scores if s.section == 'additional'), 0),
                                        'strengths': '; '.join(result.strengths[:3]),
                                        'weaknesses': '; '.join(result.weaknesses[:3]),
                                        'recommendations': '; '.join(result.recommendations[:3]),
                                        'skills_found': len(result.cv_data.skills),
                                        'experience_years': sum([exp.duration_months or 0 for exp in result.cv_data.work_experience]) / 12,
                                        'education_level': len(result.cv_data.education),
                                        'evaluation_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                                        'status': 'evaluated'
                                    }
                                    results.append(candidate_result)

                            except Exception as e:
                                st.warning(f"Failed to process {uploaded_file.name}: {e}")
                                continue

                    if results:
                        st.success(f"✅ Processed {len(results)} CVs successfully!")

                        # Show summary
                        avg_score = sum([r['overall_score'] for r in results]) / len(results)
                        top_score = max([r['overall_score'] for r in results])

                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("CVs Processed", len(results))
                        with col2:
                            st.metric("Average Score", f"{avg_score:.1f}")
                        with col3:
                            st.metric("Top Score", f"{top_score:.1f}")

                        # Show results
                        st.subheader("📊 Batch Processing Results")
                        results_df = pd.DataFrame(results)
                        st.dataframe(results_df, use_container_width=True)

                        # Export to Excel
                        processor = ExcelProcessor()
                        with tmp_module.NamedTemporaryFile(suffix='.xlsx', delete=False) as tmp_result:
                            output_path = processor.export_results_to_excel(results, tmp_result.name)

                            # Read for download
                            with open(output_path, 'rb') as f:
                                results_data = f.read()

                            st.download_button(
                                label="📥 Download Batch Results Excel",
                                data=results_data,
                                file_name=f"batch_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                            )

                            # Clean up
                            Path(output_path).unlink(missing_ok=True)

                    else:
                        st.error("❌ No CVs were successfully processed")

                    # Clean up temp files
                    for temp_file in temp_files:
                        try:
                            Path(temp_file).unlink(missing_ok=True)
                        except:
                            pass

                except Exception as e:
                    st.error(f"❌ Batch processing failed: {e}")


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
