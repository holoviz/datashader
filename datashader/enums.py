from __future__ import annotations
from enum import Enum


# Enum used to specify how the second stage aggregation is performed
# for 2-stage antialiased lines.
class AntialiasCombination(Enum):
    SUM_1AGG = 1
    SUM_2AGG = 2
    MIN = 3
    MAX = 4
    FIRST = 5
    LAST = 6
