"""
Useful functions.
"""

from collections.abc import Collection, Iterable
import csv
from datetime import datetime
import multiprocessing
import os
from pathlib import Path
import typing as tp

import capstone
from torch import nn
import pefile


def one_and_only_one(*args) -> bool:
    state = False
    for arg in args:
        if arg and state:
            return False
        if arg and not state:
            state = True
    return state


def count_parameters(model: nn.Module):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def estimate_memory_needs(
    model: nn.Module, float_type: tp.Literal["fp32", "fp16"], optimizer_type: tp.Literal["Adam", "BitsAndBytes", "SGD"]
) -> int:
    float_type_map = {"fp32": 4, "fp16": 2, "mixed": 6}
    optimizer_type_map = {"Adam": 8, "BitsAndBytes": 2, "SGD": 4}
    c = count_parameters(model)
    return c * float_type_map[float_type] + c * optimizer_type_map[optimizer_type]


def message(start: bool, ___file__: str = __file__) -> str:
    return f"{'STARTING' if start else 'FINISHING'} {os.path.basename(___file__)} @{datetime.now()}"


def get_num_workers() -> int:
    count = len(os.sched_getaffinity(0))
    # count = multiprocessing.cpu_count()
    # return max(count - 4, count // 2 + 1)
    return count


def get_highest_path(path_or_files: Collection[Path] | Path, lstrip: str = "", rstrip: str = "") -> Path:
    if isinstance(path_or_files, (Path, str)):
        files = Path(path_or_files).iterdir()
    else:
        files = path_or_files
    return list(sorted(files, key=lambda p: int(p.stem.lstrip(lstrip).rstrip(rstrip))))[-1]


def mem(path_or_files: Collection[Path] | Path) -> float:
    if isinstance(path_or_files, (Path, str)):
        files = Path(path_or_files).iterdir()
    else:
        files = path_or_files
    return sum(f.stat().st_size for f in files) * 1e-9


def clear_cache(*datasets) -> list[int]:
    return [d.clear_cache() for d in datasets]


def adjust_csv_rows(in_file: Path, out_file: Path, adjust_fn: tp.Callable, skip_header: bool = True) -> None:
    with open(in_file, "r", encoding="utf-8") as in_handle, open(out_file, "w") as out_handle:
        reader = csv.reader(in_handle)
        writer = csv.writer(out_handle)
        if skip_header:
            writer.writerow(next(reader))
        for row in reader:
            if row:
                writer.writerow(adjust_fn(row))


def get_text_section_bounds(f: Path, errors: str = "raise") -> tp.Tuple[int, int]:
    try:
        pe = pefile.PE(f.as_posix())
    except pefile.PEFormatError as e:
        if errors == "raise":
            raise e
        return None

    if not pe.FILE_HEADER.IMAGE_FILE_32BIT_MACHINE:  # filter non-x86 binaries
        return None
    if pe.FILE_HEADER.Machine == 0:  # unknown machine
        pass

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


def verify_text_section_bounds(file: Path, l: int, u: int, errors: bool) -> int:
    st_size = file.stat().st_size
    if st_size == 0:
        if errors:
            raise ValueError(f"File {file} is empty.")
        return 1
    if l >= st_size:
        if errors:
            raise ValueError(f"Lower bound {l} is greater than file size {st_size}.")
        return 2
    if u >= st_size:
        if errors:
            raise ValueError(f"Upper bound {u} is greater than file size {st_size}.")
        return 3
    if u - l < 1:
        if errors:
            raise ValueError(f"Upper bound {u} is less or equal than lower bound {l}.")
        return 4

    return 0


def read_file(file: Path, l: int = 0, u: tp.Optional[int] = None) -> bytes:
    st_size = file.stat().st_size
    u = st_size - 1 if u is None else u
    verify_text_section_bounds(file, l, u, True)
    with open(file, "rb", encoding=None) as handle:
        handle.seek(l, 0)
        binary = handle.read(u - l)
    return binary


def maybe_remove(f: Path, remove: bool) -> None:
    if remove:
        f.unlink()


def convert_size(size: tp.Union[int, float, str]) -> tp.Union[float, str]:
    symbols_to_size = {"B": 1, "K": 1024, "M": 1024 ** 2, "G": 1024 ** 3, "T": 1024 ** 4}

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


class OutputManager:
    def __init__(self, root: Path = ".") -> None:
        self.root = Path(root)
        self.data = self.root / "data"
        self.output = self.root / "output"

        # prepare paths
        self.download_sorel = self.data / "download_sorel"
        self.download_windows = self.data / "download_windows"
        self.extract = self.data / "extract"
        self.unpack = self.data / "unpack"
        self.filter = self.data / "filter"
        self.parse = self.data / "parse"
        self.disassemble = self.data / "disassemble"
        self.pre_normalized = self.data / "pre_normalized"

        # pretrain paths
        self.pretrain = self.data / "pretrain"
        self.pseudosupervised = self.data / "pseudosupervised"
        self.models = self.output / "models"
        self.encoder = self.models / "encoder"
        self.decoder = self.models / "decoder"
        # train paths
        # self.models
        self.pseudo_supervised = self.models / "pseudo_supervised"
        self.supervised = self.models / "supervised"
        self.unsupervised = self.models / "unsupervised"

        # chop paths
        self.snippets = self.data / "snippets"
        self.snippets_mal = self.snippets / "mal"
        self.snippets_ben = self.snippets / "ben"

        self.bounds_file = self.output / "bounds.csv"
        self.bounds_full_file = self.output / "bounds_full.csv"
        self.summary_file = self.output / "summary.json"
        self.tokenizers = self.output / "tokenizers"

    @property
    def prepare_paths(self) -> list[Path]:
        return [
            self.download_sorel,
            self.download_windows,
            self.extract,
            self.unpack,
            self.filter,
            self.parse,
            self.disassemble,
            self.pre_normalized,
        ]

    def mkdir_prepare_paths(self, *, exist_ok: bool = False, parents: bool = False) -> None:
        for p in self.prepare_paths:
            p.mkdir(exist_ok=exist_ok, parents=parents)

    def rmdir_prepare_paths(self, *, ignore_errors: bool = True) -> None:
        for p in self.prepare_paths:
            try:
                p.rmdir()
            except OSError as err:
                if not ignore_errors:
                    raise err
