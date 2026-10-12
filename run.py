"""
DocuMind AI: Quick Launch Entrypoint

You can run DocuMind AI either using:
    streamlit run app/streamlit_app.py
OR simply:
    python run.py
"""

import sys
import subprocess
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent

def main():
    print("==================================================")
    print("🤖 Launching DocuMind AI — Document Q&A Chatbot...")
    print("==================================================")
    
    app_path = ROOT_DIR / "app" / "streamlit_app.py"
    if not app_path.exists():
        print(f"Error: Could not locate app at {app_path}")
        sys.exit(1)

    cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(app_path),
        "--browser.gatherUsageStats=false",
    ]

    try:
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        print("\nDocuMind AI stopped.")
    except Exception as e:
        print(f"\nFailed to launch Streamlit: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
