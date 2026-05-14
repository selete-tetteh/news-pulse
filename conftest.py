# conftest.py
#
# Adds the project root to sys.path so pytest can resolve imports
# like `from src.pipeline.transform import ...` without needing
# to install the package or set PYTHONPATH manually.
#
# pytest discovers and loads this file automatically before running any tests.

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))