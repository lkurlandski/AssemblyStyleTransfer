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
    return list(sorted(files, key=lambda p: int(p.stem.lstrip(lstrip).rstrip(rstrip))))[-1]


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


class OutputManager:
    root: Path
    download: Path = Path("download")
    extract: Path = Path("extract")
    unpack: Path = Path("unpack")
    filter: Path = Path("filter")
    parse: Path = Path("parse")
    disassemble: Path = Path("disassemble")
    snippets: Path = Path("snippets")
    snippets_mal: Path = Path("mal")
    snippets_ben: Path = Path("ben")
    bounds_file: Path = Path("bounds.csv")
    summary_file: Path = Path("summary.json")
    tokenizers: Path = Path("tokenizers")
    models: Path = Path("models")
    encoder: Path = Path("encoder")
    decoder: Path = Path("decoder")
    pseudo_supervised = Path("pseudo_supervised")

    def __init__(self, root: Path = ".") -> None:
        self.root = Path(root)
        self.data = self.root / "data"
        self.output = self.root / "output"

        self.download = self.data / "download"
        self.extract = self.data / "extract"
        self.unpack = self.data / "unpack"
        self.filter = self.data / "filter"
        self.parse = self.data / "parse"
        self.disassemble = self.data / "disassemble"
        self.snippets = self.data / "snippets"
        self.snippets_mal = self.snippets / "mal"
        self.snippets_ben = self.snippets / "ben"

        self.bounds_file = self.output / "bounds.csv"
        self.summary_file = self.output / "summary.json"
        self.tokenizers = self.output / "tokenizers"
        self.models = self.output / "models"
        self.encoder = self.models / "encoder"
        self.decoder = self.models / "decoder"
        self.pseudo_supervised = self.models / "pseudo_supervised"
        self.supervised = self.models / "supervised"
        self.unsupervised = self.models / "unsupervised"
