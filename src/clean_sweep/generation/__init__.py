from .core import generate_teacher_traces, load_model_and_tokenizer
from .methods_strategic import (
    StrategicAntidistillationLogitsProcessor,
    StrategicProductOfExpertsLogitsProcessor,
)

__all__ = [
    "generate_teacher_traces",
    "load_model_and_tokenizer",
    "StrategicAntidistillationLogitsProcessor",
    "StrategicProductOfExpertsLogitsProcessor",
]
