#!/usr/bin/env python3
"""
Demo of free AI models integration for CV evaluation and chat.

Created by: Zied Boughdir (@zinzied)
GitHub: https://github.com/zinzied/cv-evaluation-system
"""

import sys
import tempfile
from pathlib import Path

# Add parent directory to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from cv_evaluator.ai.free_models import (
    list_available_models, auto_select_ai_model, get_ai_response, set_ai_model
)
from cv_evaluator.core.evaluator import CVEvaluator


def demo_available_models():
    """Demo 1: Show available free AI models."""
    print("=== Demo 1: Available Free AI Models ===")
    
    available_models = list_available_models()
    
    print("🤖 Checking available AI models...")
    
    if not any(available_models.values()):
        print("❌ No AI models are currently available")
        print("\n💡 To set up free AI models:")
        print("1. Ollama (Recommended):")
        print("   - Download: https://ollama.ai")
        print("   - Install model: ollama pull llama2")
        print("   - Start server: ollama serve")
        print("\n2. Hugging Face Transformers:")
        print("   - Install: pip install transformers torch")
        print("   - Models download automatically")
        print("\n3. LocalAI or compatible API:")
        print("   - Set up at http://localhost:8080")
        return False
    
    print("✅ Available models:")
    for model_name, is_available in available_models.items():
        status = "✅" if is_available else "❌"
        model_type = "Ollama" if "ollama" in model_name else "Hugging Face" if "hf" in model_name else "API"
        print(f"   {status} {model_name} ({model_type})")
    
    # Auto-select best model
    selected = auto_select_ai_model()
    if selected:
        print(f"\n🎯 Auto-selected model: {selected}")
        return True
    else:
        print("\n❌ Failed to auto-select model")
        return False


def demo_basic_ai_chat():
    """Demo 2: Basic AI chat functionality."""
    print("\n=== Demo 2: Basic AI Chat ===")
    
    # Test basic AI response
    test_questions = [
        "What makes a good CV?",
        "What skills are important for a software engineer?",
        "How should work experience be presented in a CV?"
    ]
    
    print("🤖 Testing basic AI responses...")
    
    for question in test_questions:
        print(f"\n❓ Question: {question}")
        try:
            response = get_ai_response(question, max_tokens=200)
            print(f"🤖 AI Response: {response[:150]}...")
        except Exception as e:
            print(f"❌ Error: {e}")
    
    return True


def demo_cv_analysis_with_ai():
    """Demo 3: CV analysis enhanced with AI."""
    print("\n=== Demo 3: AI-Enhanced CV Analysis ===")
    
    # Sample CV content
    sample_cv = """
    Alex Johnson
    alex.johnson@email.com
    +1-555-0123
    
    PROFESSIONAL SUMMARY
    Experienced software engineer with 4+ years in full-stack development.
    Passionate about clean code and agile methodologies.
    
    TECHNICAL SKILLS
    Programming: Python, JavaScript, Java, TypeScript
    Web Technologies: React, Node.js, Django, Flask
    Databases: PostgreSQL, MongoDB, Redis
    Cloud: AWS, Docker, Kubernetes
    Tools: Git, Jenkins, Jira
    
    PROFESSIONAL EXPERIENCE
    
    Senior Software Engineer | TechCorp Inc. | 2022 - Present
    • Lead development of microservices architecture
    • Mentor team of 2 junior developers
    • Improved system performance by 30%
    • Technologies: Python, Django, AWS, Docker
    
    Software Engineer | StartupXYZ | 2020 - 2022
    • Developed RESTful APIs and web applications
    • Collaborated with cross-functional teams
    • Technologies: JavaScript, React, Node.js, MongoDB
    
    EDUCATION
    Bachelor of Science in Computer Science
    University of Technology | 2020
    
    CERTIFICATIONS
    • AWS Certified Developer (2023)
    • Scrum Master Certification (2022)
    """
    
    try:
        # Create evaluator with AI enabled
        evaluator = CVEvaluator(use_ai=True)
        
        print("🔄 Processing CV with AI enhancement...")
        
        # Parse CV
        cv_data = evaluator.cv_parser.parse_cv(sample_cv)
        
        # Analyze with AI enhancement
        result = evaluator.analyzer.analyze_cv(cv_data)
        
        print("✅ AI-enhanced analysis completed!")
        print(f"   Candidate: {result.cv_data.personal_info.name}")
        print(f"   Overall Score: {result.overall_score:.1f}/100")
        print(f"   Job Fit: {result.fit_percentage:.1f}%")
        
        # Show AI-enhanced insights
        if result.strengths:
            print(f"\n💪 Strengths ({len(result.strengths)}):")
            for i, strength in enumerate(result.strengths[:3], 1):
                print(f"   {i}. {strength}")
        
        if result.recommendations:
            print(f"\n💡 Recommendations ({len(result.recommendations)}):")
            for i, rec in enumerate(result.recommendations[:3], 1):
                print(f"   {i}. {rec}")
        
        return result
        
    except Exception as e:
        print(f"❌ AI-enhanced analysis failed: {e}")
        return None


def demo_interactive_cv_chat():
    """Demo 4: Interactive chat about a CV."""
    print("\n=== Demo 4: Interactive CV Chat ===")
    
    # Use the CV from previous demo
    sample_cv = """
    Alex Johnson - Senior Software Engineer
    Skills: Python, JavaScript, React, AWS, Docker
    Experience: 4+ years in full-stack development
    Current Role: Senior Software Engineer at TechCorp Inc.
    Education: BS Computer Science, University of Technology
    Certifications: AWS Certified Developer, Scrum Master
    """
    
    # Sample questions to ask about the CV
    chat_questions = [
        "What programming languages does this candidate know?",
        "How many years of experience does Alex have?",
        "Is this candidate suitable for a senior developer role?",
        "What are the candidate's cloud computing skills?",
        "Does this candidate have leadership experience?"
    ]
    
    print("🤖 Starting interactive CV chat demo...")
    
    for question in chat_questions:
        print(f"\n❓ Question: {question}")
        
        try:
            # Create AI prompt with CV context
            prompt = f"""
You are a CV evaluation expert. Here's information about a candidate:

{sample_cv}

User question: {question}

Please provide a helpful, professional response about this candidate based on the CV information.
"""
            
            response = get_ai_response(prompt, max_tokens=300)
            print(f"🤖 AI Response: {response}")
            
        except Exception as e:
            print(f"❌ Chat error: {e}")
    
    return True


def demo_model_comparison():
    """Demo 5: Compare different AI models if available."""
    print("\n=== Demo 5: AI Model Comparison ===")
    
    available_models = list_available_models()
    available_list = [name for name, available in available_models.items() if available]
    
    if len(available_list) < 2:
        print("⚠️ Need at least 2 models for comparison")
        print(f"Available models: {available_list}")
        return
    
    test_question = "What makes a strong software engineer CV?"
    
    print(f"🔄 Testing question with different models: '{test_question}'")
    
    for model_name in available_list[:3]:  # Test up to 3 models
        print(f"\n🤖 Testing with {model_name}:")
        
        try:
            # Set specific model
            if set_ai_model(model_name):
                response = get_ai_response(test_question, max_tokens=200)
                print(f"Response: {response[:100]}...")
            else:
                print(f"❌ Failed to set model: {model_name}")
                
        except Exception as e:
            print(f"❌ Error with {model_name}: {e}")


def main():
    """Run all free AI demos."""
    print("🚀 Free AI Models Demo for CV Evaluation")
    print("Created by: Zied Boughdir (@zinzied)")
    print("=" * 60)
    
    try:
        # Demo 1: Check available models
        models_available = demo_available_models()
        
        if not models_available:
            print("\n⚠️ No AI models available. Please set up at least one free model to continue.")
            print("\nThe system will still work with basic rule-based evaluation.")
            return False
        
        # Demo 2: Basic AI chat
        demo_basic_ai_chat()
        
        # Demo 3: AI-enhanced CV analysis
        analysis_result = demo_cv_analysis_with_ai()
        
        # Demo 4: Interactive CV chat
        demo_interactive_cv_chat()
        
        # Demo 5: Model comparison (if multiple models available)
        demo_model_comparison()
        
        print("\n" + "=" * 60)
        print("🎉 Free AI Models Demo Completed!")
        
        print("\n🚀 How to use free AI features:")
        print("1. Command Line Chat:")
        print("   python -m cv_evaluator chat cv.pdf")
        
        print("\n2. List Available Models:")
        print("   python -m cv_evaluator list-ai-models")
        
        print("\n3. Web Interface with AI Chat:")
        print("   python run_web_app.py")
        print("   (Select 'AI Chat' mode)")
        
        print("\n4. Enhanced Evaluation:")
        print("   python -m cv_evaluator evaluate cv.pdf --ai-enhanced")
        
        print("\n💡 Free AI models provide:")
        print("   • Enhanced CV insights and recommendations")
        print("   • Interactive chat about candidates")
        print("   • Natural language explanations")
        print("   • Contextual analysis and suggestions")
        
        print("\n🔧 Supported free models:")
        print("   • Ollama (llama2, mistral, codellama)")
        print("   • Hugging Face Transformers (local)")
        print("   • LocalAI and compatible APIs")
        print("   • No API keys or paid services required!")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
