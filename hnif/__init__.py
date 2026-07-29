"""
Hybrid Neural Interpretability Framework (HNIF)
A model-agnostic middleware for extracting, filtering, and translating
Transformer-based classification decisions into human-readable audits.
"""

# Expose the main entry point to the user
from .adapter import run_hnif_analysis

__version__ = "1.0.0"
__author__ = "Dinitha Fernando"
