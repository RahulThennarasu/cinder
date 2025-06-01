# cinder/__init__.py
"""
Cinder - ML model debugging and analysis dashboard
"""
__version__ = "1.8.3"

from backend.model_interface.connector import ModelDebugger
from backend.ml_analysis.bit_assistant import BitAssistant

__all__ = ["ModelDebugger", "BitAssistant"]