"""
Run the explanation algorithm.
"""

import os

from utils import add_prefix_to_csv


def fn(row: list) -> list:
    root = os.path.dirname(os.path.realpath(__file__))
    name = row[0]
    row[0] = os.path.join(root, name)
    return row


if __name__ == "__main__":
    add_prefix_to_csv(
    '/home/lk3591/Documents/code/HMCST/data/bounds.csv',
    '/home/lk3591/Documents/code/HMCST/data/bounds_full.csv',
    fn,
)
