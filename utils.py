"""
Useful functions.
"""

from collections.abc import Collection, Iterable
import csv
from pathlib import Path
import typing as tp

import capstone
import pefile


def get_highest_path(
    path_or_files: Collection[Path] | Path,
    lstrip: str = "",
    rstrip: str = "",
) -> Path:
    if isinstance(path_or_files, (Path, str)):
        files = Path(path_or_files).iterdir()
    else:
        files = path_or_files
    return list(sorted(files, key=lambda p: int(p.stem.lstrip(lstrip).rstrip())))[-1]


def mem(path_or_files: Collection[Path] | Path) -> float:
    if isinstance(path_or_files, (Path, str)):
        files = Path(path_or_files).iterdir()
    else:
        files = path_or_files
    return sum(f.stat().st_size for f in files) * 10e-9


def clear_cache(*datasets) -> list[int]:
    return [d.clear_cache() for d in datasets]


def adjust_csv_rows(in_file: Path, out_file: Path, adjust_fn: tp.Callable, skip_header: bool = True) -> None:
    with open(in_file, "r", encoding="utf-8") as in_handle, open(out_file, "w") as out_handle:
        reader = csv.reader(in_handle)
        writer = csv.writer(out_handle)
        if skip_header:
            writer.writerow(next(reader))
        for row in reader:
            writer.writerow(adjust_fn(row))


def get_text_section_bounds(f: Path, errors: str = "raise") -> tp.Tuple[int, int]:
    try:
        pe = pefile.PE(f.as_posix())
    except pefile.PEFormatError as e:
        if errors == "raise":
            raise e
        return None

    for section in pe.sections:
        if ".text" in section.Name.decode("utf-8", errors="ignore"):
            lower = section.PointerToRawData
            upper = lower + section.SizeOfRawData
            return lower, upper

    if errors == "raise":
        raise pefile.PEFormatError("No .text section found.")
    return None


def instruction_as_str(ins: list, address: bool = False) -> str:
    a, _, m, o = ins
    r = hex(a) + "\t" if address else ""
    r += f"{m} {o}"
    return r


def disasm(md: capstone.Cs, code: bytes, format_fn: tp.Callable = tuple, start: int = 0x0) -> list:
    return [format_fn(i) for i in md.disasm_lite(code, start)]


def read_file(file: Path, l: tp.Optional[int] = None, u: tp.Optional[int] = None) -> bytes:
    with open(file, "rb", encoding=None) as handle:
        handle.seek(l)
        binary = handle.read(u)
    return binary


def convert_size(size: tp.Union[int, float, str]) -> tp.Union[float, str]:
    symbols_to_size = {"B": 1, "K": 1024, "M": 1024**2, "G": 1024**3, "T": 1024**4}

    if isinstance(size, (int, float)):
        s_, m_ = "B", 1
        for s, m in symbols_to_size.items():
            if m >= size:
                break
            s_, m_ = s, m

        return f"{round(size / m_, 2)}{s_}"

    if isinstance(size, str):
        for s, d in symbols_to_size.items():
            if size.replace(s, "").isdigit():
                return float(size.replace(s, "")) * d

    raise TypeError()


def files_in_dir(path: Path) -> int:
    return sum(1 for _ in path.iterdir())
