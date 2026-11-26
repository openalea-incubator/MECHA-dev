import os
import sys

# A non-interactive matplotlib backend: the frozen legacy tests and examples show their figures
os.environ.setdefault("MPLBACKEND", "Agg")
# MKL's strict conditional numerical reproducibility, set before numpy loads MKL: its default kernels' results depend on
# the arrays' memory alignment (SuperLU's solves in the Kron reductions call BLAS), so a kind evaluated in a batch and
# alone, or in a thread and in one, would differ in the last bits and the bit-for-bit tests fail (the Windows CI)
os.environ.setdefault("MKL_CBWR", "AUTO,STRICT")
if "matplotlib" in sys.modules:
    import matplotlib
    matplotlib.use("Agg")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))          # plants.py, for every test folder
