"""

"""

from transformers import EarlyStoppingCallback, ProgressCallback, TrainerCallback


def get_callbacks(patience: int = 5, threshold: int = 0.000) -> list[TrainerCallback]:
    return [EarlyStoppingCallback(patience, threshold)]
