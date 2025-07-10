"""
Chat interface for CV evaluation system.

Created by: Zied Boughdir (@zinzied)
GitHub: https://github.com/zinzied/cv-evaluation-system
"""

import streamlit as st
import tempfile
from pathlib import Path
from typing import Optional, List, Dict, Any
import json

from .core.evaluator import CVEvaluator
from .ai.free_models import get_ai_response, list_available_models, auto_select_ai_model, set_ai_model
from .core.models import CVData


class CVChatInterface:
    """Chat interface for CV evaluation and analysis."""
    
    def __init__(self):
        self.evaluator = None
        self.current_cv_data = None
        self.chat_history = []
    
    def render_chat_interface(self):
        """Render the chat interface in Streamlit."""
        st.header("💬 CV Evaluation Chat Assistant")
        st.markdown("Upload a CV and chat with AI about the candidate's qualifications")
        
        # Model selection
        self._render_model_selection()
        
        # CV upload section
        self._render_cv_upload()
        
        # Chat section
        if self.current_cv_data:
            self._render_chat_section()
        else:
            st.info("👆 Please upload a CV to start chatting about the candidate")
    
    def _render_model_selection(self):
        """Render AI model selection interface."""
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
                return
            
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
    
    def _render_cv_upload(self):
        """Render CV upload interface."""
        st.subheader("📄 Upload CV")
        
        uploaded_file = st.file_uploader(
            "Choose a CV file",
            type=['pdf', 'txt'],
            help="Upload a PDF or text file containing the CV"
        )
        
        if uploaded_file is not None:
            # Process the uploaded CV
            with st.spinner("Processing CV..."):
                success = self._process_uploaded_cv(uploaded_file)
                
                if success:
                    st.success(f"✅ CV processed: {self.current_cv_data.personal_info.name or 'Unknown candidate'}")
                    
                    # Show CV summary
                    with st.expander("📋 CV Summary"):
                        self._display_cv_summary()
                else:
                    st.error("❌ Failed to process CV")
    
    def _process_uploaded_cv(self, uploaded_file) -> bool:
        """Process uploaded CV file."""
        try:
            # Save file temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_file_path = tmp_file.name
            
            # Create evaluator and process CV
            self.evaluator = CVEvaluator(use_ai=True)
            
            if uploaded_file.name.endswith('.pdf'):
                # Extract from PDF
                extraction_result = self.evaluator.pdf_extractor.extract_text(tmp_file_path)
                cv_text = extraction_result.get('text', '')
            else:
                # Read text file
                cv_text = Path(tmp_file_path).read_text(encoding='utf-8')
            
            # Parse CV
            self.current_cv_data = self.evaluator.cv_parser.parse_cv(cv_text)
            
            # Clear chat history when new CV is uploaded
            self.chat_history = []
            
            # Clean up
            Path(tmp_file_path).unlink(missing_ok=True)
            
            return True
            
        except Exception as e:
            st.error(f"Error processing CV: {e}")
            return False
    
    def _display_cv_summary(self):
        """Display a summary of the processed CV."""
        if not self.current_cv_data:
            return
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Personal Information:**")
            if self.current_cv_data.personal_info.name:
                st.write(f"Name: {self.current_cv_data.personal_info.name}")
            if self.current_cv_data.personal_info.email:
                st.write(f"Email: {self.current_cv_data.personal_info.email}")
            
            st.write(f"**Skills Found:** {len(self.current_cv_data.skills)}")
            if self.current_cv_data.skills:
                skills_preview = ", ".join([skill.name for skill in self.current_cv_data.skills[:5]])
                st.write(f"Top skills: {skills_preview}...")
        
        with col2:
            st.write(f"**Work Experience:** {len(self.current_cv_data.work_experience)} entries")
            if self.current_cv_data.work_experience:
                recent_job = self.current_cv_data.work_experience[0]
                st.write(f"Recent: {recent_job.position} at {recent_job.company}")
            
            st.write(f"**Education:** {len(self.current_cv_data.education)} entries")
            if self.current_cv_data.education:
                recent_edu = self.current_cv_data.education[0]
                st.write(f"Latest: {recent_edu.degree} from {recent_edu.institution}")
    
    def _render_chat_section(self):
        """Render the chat interface."""
        st.subheader("💬 Chat about this CV")
        
        # Display chat history
        if self.chat_history:
            st.write("**Chat History:**")
            for i, (question, answer) in enumerate(self.chat_history):
                with st.container():
                    st.write(f"**You:** {question}")
                    st.write(f"**Assistant:** {answer}")
                    st.divider()
        
        # Quick question buttons
        st.write("**Quick Questions:**")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("🎯 Overall Assessment"):
                self._ask_question("What is your overall assessment of this candidate?")
        
        with col2:
            if st.button("💪 Key Strengths"):
                self._ask_question("What are the key strengths of this candidate?")
        
        with col3:
            if st.button("📈 Areas for Improvement"):
                self._ask_question("What areas should this candidate improve?")
        
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
                    self._ask_question(user_question)
                    st.rerun()
        
        with col2:
            if st.button("🗑️ Clear Chat"):
                self.chat_history = []
                st.rerun()
    
    def _ask_question(self, question: str):
        """Process a user question about the CV."""
        if not self.current_cv_data:
            return
        
        try:
            # Get AI response
            if hasattr(self.evaluator.analyzer, 'chat_about_cv'):
                answer = self.evaluator.analyzer.chat_about_cv(self.current_cv_data, question)
            else:
                # Fallback to general AI response
                cv_summary = self._get_cv_context()
                prompt = f"""
CV Information:
{cv_summary}

User Question: {question}

Please provide a helpful response about this candidate based on their CV.
"""
                answer = get_ai_response(prompt, max_tokens=500)
            
            # Add to chat history
            self.chat_history.append((question, answer))
            
        except Exception as e:
            error_msg = f"Sorry, I encountered an error: {e}"
            self.chat_history.append((question, error_msg))
    
    def _get_cv_context(self) -> str:
        """Get CV context for AI prompts."""
        if not self.current_cv_data:
            return ""
        
        context_parts = []
        
        # Personal info
        if self.current_cv_data.personal_info.name:
            context_parts.append(f"Candidate: {self.current_cv_data.personal_info.name}")
        
        # Skills
        if self.current_cv_data.skills:
            skills = ", ".join([skill.name for skill in self.current_cv_data.skills[:10]])
            context_parts.append(f"Skills: {skills}")
        
        # Experience
        if self.current_cv_data.work_experience:
            exp_summary = []
            for exp in self.current_cv_data.work_experience[:3]:
                exp_summary.append(f"{exp.position} at {exp.company}")
            context_parts.append(f"Experience: {'; '.join(exp_summary)}")
        
        # Education
        if self.current_cv_data.education:
            edu_summary = []
            for edu in self.current_cv_data.education[:2]:
                edu_summary.append(f"{edu.degree} from {edu.institution}")
            context_parts.append(f"Education: {'; '.join(edu_summary)}")
        
        return "\n".join(context_parts)


def render_chat_interface():
    """Main function to render the chat interface."""
    if 'chat_interface' not in st.session_state:
        st.session_state.chat_interface = CVChatInterface()
    
    st.session_state.chat_interface.render_chat_interface()
