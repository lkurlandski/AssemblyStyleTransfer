"""
Run the explanation algorithm.
"""

import os

from utils import adjust_csv_rows, OutputManager


root = os.path.dirname(os.path.realpath(__file__))


def fn(row: list) -> list:
    name = row[0]
    row[0] = os.path.join(root, name)
    return row


def main() -> None:
    oh = OutputManager()
    adjust_csv_rows(oh.bounds_file.as_posix(), oh.bounds_full_file.as_posix(), fn)


if __name__ == "__main__":
    main()
