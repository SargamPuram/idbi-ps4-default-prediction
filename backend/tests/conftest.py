"""
Ensures `backend/` (the directory containing the `ml` and `app` packages) is
importable regardless of the working directory pytest is invoked from, e.g.
both of these should work:

    cd backend && venv/Scripts/python -m pytest tests/ -v
    venv/Scripts/python.exe -m pytest backend/tests/ -v   (from repo root)
"""

import os
import sys

BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)
