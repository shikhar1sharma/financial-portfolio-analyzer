"""
Main Project Runner
Execute this file to verify the project structure
"""

import os


def verify_project_structure():
    """Verify that all project files were created"""
    required_files = [
        "requirements.txt",
        "README.md",
        ".env.example",
        "mcp_servers/market_data_server.py",
        "mcp_servers/portfolio_server.py",
        "agents/financial_agent.py",
        "dashboard/main_dashboard.py",
        "scripts/start_all_services.py",
        "config/mcp_config.json",
    ]

    missing_files = []
    for file in required_files:
        if not os.path.exists(file):
            missing_files.append(file)

    if missing_files:
        print("❌ Missing files:")
        for file in missing_files:
            print(f"   - {file}")
        return False
    else:
        print("✅ All required files are present!")
        return True


if __name__ == "__main__":
    print("🚀 Financial Portfolio Analyzer Project Verification")
    print("=" * 50)

    if verify_project_structure():
        print()
        print("📋 Next Steps:")
        print("1. pip install -r requirements.txt")
        print("2. cp .env.example .env")
        print("3. Edit .env with your API keys")
        print("4. Start implementing the TODO items!")
        print()
        print("🎯 Begin with: mcp_servers/market_data_server.py")
        print("💡 Each TODO comment explains exactly what to implement")
    else:
        print("❌ Project structure verification failed!")
