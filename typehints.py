from pathlib import PurePath
from typing import Dict, Union

FName = Union[str, PurePath]
Numeric = Union[int, float]
StateDict = Union[Dict, None]
MetricDict = Dict[Numeric, float]
