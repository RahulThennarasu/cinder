# Update __init__.py to expose BitOptimizer
from backend.ml_analysis.code_generator import SimpleCodeGenerator
from backend.ml_analysis.improvement_advisor import ModelImprovementAdvisor
from backend.ml_analysis.bit_assistant import BitOptimizer

__all__ = ['SimpleCodeGenerator', 'ModelImprovementAdvisor', 'BitOptimizer']