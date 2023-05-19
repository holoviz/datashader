from enum import Enum


class DataLibrary(Enum):
    PandasDF = 1
    DaskDF = 2
    CuDF = 3
    DaskCuDF = 4
