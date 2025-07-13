#!/usr/bin/env python3
"""
Check API configuration and test connections.
"""

import os
import sys
from pathlib import Path

def print_header():
    """Print header."""
    print("🔍 CV Evaluator - Configuration Checker")
    print("=" * 50)

def check_env_file():
    """Check if .env file exists and show content."""
    env_file = Path('.env')
    
    print("📁 Checking .env file...")
    if env_file.exists():
        print("✅ .env file found")
        
        # Read and display non-empty, non-comment lines
        content = env_file.read_text()
        lines = [line.strip() for line in content.splitlines() 
                if line.strip() and not line.strip().startswith('#')]
        
        if lines:
            print("\n📋 Configuration found:")
            for line in lines:
                if '=' in line:
                    key, value = line.split('=', 1)
                    if value.strip():
                        print(f"  ✅ {key}: {'*' * min(len(value), 10)}...")
                    else:
                        print(f"  ❌ {key}: (empty)")
        else:
            print("⚠️ .env file is empty or contains only comments")
        
        return True
    else:
        print("❌ .env file not found")
        print("\n💡 To create .env file:")
        print("  • Run: create_env.bat")
        print("  • Or copy: .env.template to .env")
        print("  • Or create manually with your API keys")
        return False

def check_environment_variables():
    """Check environment variables."""
    print("\n🌍 Checking environment variables...")
    
    api_keys = {
        'GROQ_API_KEY': 'Groq API (Fast & Free)',
        'OPENAI_API_KEY': 'OpenAI API (Premium)',
        'ANTHROPIC_API_KEY': 'Anthropic Claude API',
        'GOOGLE_API_KEY': 'Google Gemini API',
        'HUGGINGFACE_API_KEY': 'Hugging Face API'
    }
    
    found_keys = 0
    for key, description in api_keys.items():
        value = os.getenv(key)
        if value:
            print(f"  ✅ {key}: {description}")
            found_keys += 1
        else:
            print(f"  ❌ {key}: {description} (not set)")
    
    print(f"\n📊 Summary: {found_keys}/{len(api_keys)} API keys configured")
    return found_keys > 0

def test_api_config():
    """Test API configuration using the system."""
    print("\n🧪 Testing API configuration...")
    
    try:
        # Add current directory to path
        sys.path.insert(0, str(Path(__file__).parent))
        
        from cv_evaluator.utils.api_config import api_config
        
        enabled_providers = api_config.get_enabled_providers()
        
        if enabled_providers:
            print("✅ API configuration system loaded successfully")
            print("\n🔑 Enabled providers:")
            for provider, config in enabled_providers.items():
                print(f"  ✅ {provider.title()}")
        else:
            print("❌ No API providers are enabled")
            
        return len(enabled_providers) > 0
        
    except Exception as e:
        print(f"❌ Error loading API configuration: {e}")
        return False

def test_ai_models():
    """Test AI model availability."""
    print("\n🤖 Testing AI models...")
    
    try:
        from cv_evaluator.ai.free_models import list_available_models
        
        available_models = list_available_models()
        available_count = sum(1 for available in available_models.values() if available)
        
        if available_count > 0:
            print(f"✅ {available_count} AI models available")
            
            # Show available models by category
            categories = {
                'API Models': [k for k, v in available_models.items() if v and any(k.startswith(p) for p in ['groq_', 'openai_', 'anthropic_', 'google_'])],
                'Local Models': [k for k, v in available_models.items() if v and k.startswith('ollama_')],
                'Other Models': [k for k, v in available_models.items() if v and not any(k.startswith(p) for p in ['groq_', 'openai_', 'anthropic_', 'google_', 'ollama_'])]
            }
            
            for category, models in categories.items():
                if models:
                    print(f"\n  {category}:")
                    for model in models[:3]:  # Show first 3
                        print(f"    ✅ {model}")
                    if len(models) > 3:
                        print(f"    ... and {len(models) - 3} more")
        else:
            print("❌ No AI models available")
            
        return available_count > 0
        
    except Exception as e:
        print(f"❌ Error testing AI models: {e}")
        return False

def provide_recommendations():
    """Provide setup recommendations."""
    print("\n💡 Recommendations")
    print("-" * 50)
    
    # Check what's missing and provide specific advice
    has_env = Path('.env').exists()
    has_api_keys = any(os.getenv(key) for key in ['GROQ_API_KEY', 'OPENAI_API_KEY', 'ANTHROPIC_API_KEY'])
    
    if not has_env and not has_api_keys:
        print("🚀 Quick setup (choose one):")
        print("\n1. **Groq API** (Fastest, Free):")
        print("   • Get key: https://console.groq.com")
        print("   • Run: create_env.bat")
        print("   • Add: GROQ_API_KEY=your_key")
        
        print("\n2. **Ollama** (Local, Private):")
        print("   • Download: https://ollama.ai")
        print("   • Run: ollama pull llama3")
        print("   • Run: ollama serve")
        
    elif has_env and not has_api_keys:
        print("📝 You have .env file but no API keys set")
        print("• Edit .env file and add your API keys")
        print("• Get Groq key (free): https://console.groq.com")
        
    elif has_api_keys:
        print("✅ You have API keys configured!")
        print("• Start the app: python run_web_app.py")
        print("• Go to 'AI Chat' mode to test")

def main():
    """Main function."""
    print_header()
    
    # Run all checks
    checks = [
        ("Environment file", check_env_file),
        ("Environment variables", check_environment_variables),
        ("API configuration", test_api_config),
        ("AI models", test_ai_models)
    ]
    
    results = []
    for name, check_func in checks:
        try:
            result = check_func()
            results.append((name, result))
        except Exception as e:
            print(f"❌ Error in {name}: {e}")
            results.append((name, False))
    
    # Summary
    print("\n📊 Configuration Summary")
    print("=" * 50)
    
    passed = 0
    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} {name}")
        if result:
            passed += 1
    
    print(f"\n🎯 Overall: {passed}/{len(results)} checks passed")
    
    # Provide recommendations
    provide_recommendations()
    
    if passed >= 2:
        print("\n🎉 Your system is ready to use!")
        print("Run: python run_web_app.py")
    else:
        print("\n🔧 Please follow the recommendations above to complete setup")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 Check cancelled by user")
    except Exception as e:
        print(f"\n❌ Check failed: {e}")
        sys.exit(1)
