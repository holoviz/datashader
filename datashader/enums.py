from __future__ import annotations
from enum import Enum

# This Enum should eventually be replaced with attributes
# and/or member functions of Reduction classes.
class AntialiasCombination(Enum):
    NONE = 0
    SUM_1AGG = 1
    SUM_2AGG = 2
    MIN = 3
    MAX = 4
    