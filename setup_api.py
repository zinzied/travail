#!/usr/bin/env python3
"""
Interactive API setup script for CV Evaluator.
Helps users configure AI model APIs easily.
"""

import os
import sys
from pathlib import Path

def print_header():
    """Print welcome header."""
    print("🤖 CV Evaluator - AI API Setup")
    print("=" * 50)
    print("This script will help you configure AI model APIs")
    print("for the CV Evaluator system.\n")

def check_existing_config():
    """Check for existing configuration."""
    env_file = Path('.env')
    if env_file.exists():
        print("📁 Found existing .env file")
        return True
    return False

def create_env_file():
    """Create .env file from template."""
    template_file = Path('.env.template')
    env_file = Path('.env')
    
    if template_file.exists():
        # Copy template to .env
        content = template_file.read_text()
        env_file.write_text(content)
        print("✅ Created .env file from template")
        return True
    else:
        print("❌ .env.template not found")
        return False

def setup_groq_api():
    """Setup Groq API configuration."""
    print("\n🚀 Setting up Groq API (Recommended - Fast & Free)")
    print("-" * 50)
    print("1. Visit: https://console.groq.com")
    print("2. Create a free account")
    print("3. Generate an API key")
    print("4. Copy the API key\n")
    
    api_key = input("Enter your Groq API key (or press Enter to skip): ").strip()
    
    if api_key:
        update_env_file('GROQ_API_KEY', api_key)
        print("✅ Groq API key saved!")
        return True
    else:
        print("⏭️ Skipping Groq API setup")
        return False

def setup_openai_api():
    """Setup OpenAI API configuration."""
    print("\n💎 Setting up OpenAI API (Premium)")
    print("-" * 50)
    print("1. Visit: https://platform.openai.com")
    print("2. Create an account and add payment method")
    print("3. Generate an API key")
    print("4. Copy the API key\n")
    
    api_key = input("Enter your OpenAI API key (or press Enter to skip): ").strip()
    
    if api_key:
        update_env_file('OPENAI_API_KEY', api_key)
        print("✅ OpenAI API key saved!")
        return True
    else:
        print("⏭️ Skipping OpenAI API setup")
        return False

def setup_anthropic_api():
    """Setup Anthropic API configuration."""
    print("\n🧠 Setting up Anthropic Claude API")
    print("-" * 50)
    print("1. Visit: https://console.anthropic.com")
    print("2. Create an account")
    print("3. Generate an API key")
    print("4. Copy the API key\n")
    
    api_key = input("Enter your Anthropic API key (or press Enter to skip): ").strip()
    
    if api_key:
        update_env_file('ANTHROPIC_API_KEY', api_key)
        print("✅ Anthropic API key saved!")
        return True
    else:
        print("⏭️ Skipping Anthropic API setup")
        return False

def setup_google_api():
    """Setup Google API configuration."""
    print("\n🌟 Setting up Google Gemini API")
    print("-" * 50)
    print("1. Visit: https://makersuite.google.com")
    print("2. Create an account")
    print("3. Generate an API key")
    print("4. Copy the API key\n")
    
    api_key = input("Enter your Google API key (or press Enter to skip): ").strip()
    
    if api_key:
        update_env_file('GOOGLE_API_KEY', api_key)
        print("✅ Google API key saved!")
        return True
    else:
        print("⏭️ Skipping Google API setup")
        return False

def setup_ollama():
    """Setup Ollama local models."""
    print("\n🏠 Setting up Ollama (Local & Private)")
    print("-" * 50)
    print("1. Download Ollama from: https://ollama.ai")
    print("2. Install Ollama on your system")
    print("3. Open terminal and run: ollama pull llama3")
    print("4. Start server: ollama serve\n")
    
    choice = input("Have you installed and started Ollama? (y/n): ").strip().lower()
    
    if choice == 'y':
        print("✅ Ollama setup noted!")
        return True
    else:
        print("ℹ️ You can set up Ollama later for local AI models")
        return False

def update_env_file(key: str, value: str):
    """Update .env file with new key-value pair."""
    env_file = Path('.env')
    
    if not env_file.exists():
        env_file.write_text("")
    
    # Read current content
    lines = env_file.read_text().splitlines()
    
    # Update or add the key
    updated = False
    for i, line in enumerate(lines):
        if line.startswith(f"{key}="):
            lines[i] = f"{key}={value}"
            updated = True
            break
    
    if not updated:
        lines.append(f"{key}={value}")
    
    # Write back to file
    env_file.write_text('\n'.join(lines) + '\n')

def test_configuration():
    """Test the API configuration."""
    print("\n🧪 Testing Configuration")
    print("-" * 50)
    
    try:
        # Add current directory to Python path
        sys.path.insert(0, str(Path(__file__).parent))
        
        from cv_evaluator.utils.api_config import api_config
        
        providers = ['groq', 'openai', 'anthropic', 'google', 'ollama']
        configured_count = 0
        
        for provider in providers:
            is_enabled = api_config.is_provider_enabled(provider)
            status = "✅ Configured" if is_enabled else "❌ Not configured"
            print(f"• {provider.title()}: {status}")
            if is_enabled:
                configured_count += 1
        
        print(f"\n📊 Summary: {configured_count}/{len(providers)} providers configured")
        
        if configured_count > 0:
            print("🎉 Great! You have at least one AI provider configured.")
        else:
            print("⚠️ No providers configured. Please set up at least one API.")
        
        return configured_count > 0
        
    except Exception as e:
        print(f"❌ Error testing configuration: {e}")
        return False

def main():
    """Main setup function."""
    print_header()
    
    # Check for existing config
    has_existing = check_existing_config()
    
    if not has_existing:
        print("📝 Creating .env configuration file...")
        if not create_env_file():
            print("❌ Could not create .env file. Please create it manually.")
            return
    
    print("\n🔧 Choose which AI providers to configure:")
    print("(You can set up multiple providers)")
    
    # Setup options
    setup_functions = [
        ("Groq API (Fastest, Free)", setup_groq_api),
        ("OpenAI API (Premium)", setup_openai_api),
        ("Anthropic Claude API", setup_anthropic_api),
        ("Google Gemini API", setup_google_api),
        ("Ollama (Local)", setup_ollama)
    ]
    
    configured_any = False
    
    for name, setup_func in setup_functions:
        choice = input(f"\nSetup {name}? (y/n): ").strip().lower()
        if choice == 'y':
            if setup_func():
                configured_any = True
    
    # Test configuration
    if configured_any:
        test_configuration()
    
    # Final instructions
    print("\n🚀 Setup Complete!")
    print("-" * 50)
    print("Next steps:")
    print("1. Start the application: python run_web_app.py")
    print("2. Go to 'AI Chat' mode")
    print("3. Check 'AI Model Settings' to verify configuration")
    print("4. Upload a document and start chatting!")
    
    if not configured_any:
        print("\n💡 Quick start options:")
        print("• For fastest setup: Get Groq API key (free)")
        print("• For privacy: Install Ollama (local)")
        print("• For premium: Get OpenAI API key")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 Setup cancelled by user")
    except Exception as e:
        print(f"\n❌ Setup failed: {e}")
        sys.exit(1)
