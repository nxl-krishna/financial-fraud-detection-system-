"""
Entry point for Streamlit Community Cloud.
Automatically points to app/streamlit_app.py.
"""
import sys
from pathlib import Path

# Ensure project root is in sys.path
ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# Execute main Streamlit dashboard
import runpy
runpy.run_path(str(ROOT_DIR / "app" / "streamlit_app.py"), run_name="__main__")
