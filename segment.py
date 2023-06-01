"""
Decide on the bounds to perform the explanations upon.
"""

from pathlib import Path
from abc import ABC, abstractmethod

import pandas as pd


class Segmenter(ABC):
    
    def __init__(self, bounds_file: Path, binaries: list[Path], attributions: list[Path]) -> None:
        
        self.bounds_file = bounds_file
        self.binaries = binaries
        self.attributions = attributions
    
    @abstractmethod
    def __call__(self) -> None:
        ...


class ChunkSegmenter(Segmenter):
    ...


class FunctionSegmenter(Segmenter):
    ...
    

class BasicBlockSegmenter(Segmenter):
    ...


if __name__ == "__main__":
    pass
