"""Temporal TinyHOOD reference, data contracts and reproducible experiments."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SIBLING = ROOT.parent / "vulkan-gnn-poc"
if str(SIBLING) not in sys.path:
    sys.path.insert(0, str(SIBLING))
