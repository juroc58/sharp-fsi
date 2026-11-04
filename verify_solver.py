"""verify_solver.py has moved to scripts/verify_solver.py.

Run:  python3 scripts/verify_solver.py
"""
import sys

sys.path.insert(0, "scripts")
from verify_solver import main

if __name__ == "__main__":
    ok = main()
    sys.exit(0 if ok else 1)
