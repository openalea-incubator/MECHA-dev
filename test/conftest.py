import os
import sys

# A non-interactive matplotlib backend: the frozen legacy tests and examples show their figures
os.environ.setdefault("MPLBACKEND", "Agg")
if "matplotlib" in sys.modules:
    import matplotlib
    matplotlib.use("Agg")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))          # plants.py, for every test folder
