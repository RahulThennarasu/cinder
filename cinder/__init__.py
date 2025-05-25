# cinder/__init__.py
"""
Cinder - ML model debugging and analysis dashboard
"""
__version__ = "1.0.7"

from backend.model_interface.connector import ModelDebugger

__all__ = ["ModelDebugger"]