"""
Acquire data for learning, including downloading and disassembling (Network-intensive & CPU intensive).

This script is a little complicated. It is intended to be run using stupid parallelization:
    - One worker should be responsible for downloading, eg,
        python prepare.py --_32_bit --responsibility=None
    - Subsequent workers should be responsible for disassembling select files that begin with `responsibility`
        python prepare.py --_32_bit --no_download --responsibility=000
        python prepare.py --_32_bit --no_download --responsibility=001
        python prepare.py --_32_bit --no_download --responsibility=002
        ...
        python prepare.py --_32_bit --no_download --responsibility=00f
"""

from argparse import ArgumentParser
from collections.abc import Collection
from copy import deepcopy
import csv
from dataclasses import dataclass
from collections import defaultdict
from itertools import chain
import os
import json
from multiprocessing import Pool
from pathlib import Path
import pefile as pe
from pprint import pprint
from random import shuffle
import re
import shutil
import signal
import subprocess
import sys
import time
import typing as tp
import warnings
import zlib

try:
    from tqdm import tqdm
except ImportError:

    def tqdm(iterable, *args, **kwargs):
        return iterable


import r2pipe

import cfg
from tokenization import get_pre_normalizer
from utils import OutputManager


N_FILES = 25
MAX_LEN = 1e6
_16_BIT = False
_32_BIT = True
_64_BIT = False
WAIT = 5

WARNS = [("Functions with same addresses detected", False), ("Function addresses improperly parsed", False)]

CHUNK = 1024
NORMALIZER = get_pre_normalizer()
ALREADY_DONE = set()


def pre_normalize(f: Path, dest_path: Path) -> Path:
    d_out = dest_path / f.parent.name
    d_out.mkdir(exist_ok=True)
    f_out = d_out / f.name
    with open(f) as handle:
        out_str = NORMALIZER.normalize_str(handle.read())
    with open(f_out, "w") as handle:
        handle.write(out_str)
    f.unlink()
    return f_out


class PDRParser:
    def __init__(self, f: Path):
        self.idx = 0
        r2 = r2pipe.open(f.as_posix())
        r2.cmd("aaaa")
        self.output = r2.cmd("pdr @@f").split("\n")

    def __iter__(self):
        return self

    def __len__(self):
        return len(self.output)

    def __next__(self):
        if self.idx == len(self):
            raise StopIteration
        func = []
        add = False
        look_for_start = False
        start = None
        end = None
        while self.idx < len(self):
            line = self.output[self.idx]
            self.idx += 1
            if not line:
                continue
            if look_for_start and start is None:
                try:
                    start = int(self._extract_address(line), 16)
                    look_for_start = False
                except ValueError:
                    start = None
            if self._indicates_end(line) and start:
                try:
                    end = int(self._extract_address(line), 16)
                except ValueError:
                    end = None
                if not self._indicates_start(line):
                    func.append(line)
                else:
                    self.idx -= 1
                return start, end, func
            if self._indicates_start(line):
                if add:
                    try:
                        end = int(self._extract_address(line), 16)
                    except ValueError:
                        end = None
                    self.idx -= 1
                    return start, end, func
                add = True
                look_for_start = True
                start = None
            if add:
                func.append(line)
        if func:
            return start, end, func
        raise StopIteration

    @staticmethod
    def _extract_address(line: str) -> str:
        return line[2:13]

    @staticmethod
    def _indicates_end(line: str) -> bool:
        if line[0] in ("└", "┌", "├"):
            return True
        return False

    @staticmethod
    def _indicates_start(line: str) -> bool:
        if line[0] in ("┌", "├"):
            return True
        return False


def disassemble(f: Path, dest_path: Path) -> list[Path]:
    outpath = dest_path / f.stem
    outpath.mkdir()
    parser = PDRParser(f)
    for i, (start, end, func) in enumerate(parser):
        f_out = outpath / f"{start}_{end}.asm"
        if f_out.exists():
            warnings.warn(f"{WARNS[0][0]} @{i=} {f=}")
            if WARNS[0][1]:
                continue
        if start is None or end is None:
            warnings.warn(f"{WARNS[1][0]} @{i=} {f=}")
            if WARNS[1][1]:
                continue
        with open(f_out, "w") as handle:
            handle.write("\n".join(func))
    return list(outpath.iterdir())


def filter_(
    f: Path,
    dest_path: Path,
    max_len: int = MAX_LEN,
    _16_bit: bool = _16_BIT,
    _32_bit: bool = _32_BIT,
    _64_bit: bool = _64_BIT,
) -> tuple[Path, int]:
    def ret(keep: bool):
        f_out = dest_path / f.name
        if keep:
            f.rename(f_out)
        else:
            f.unlink()
        return f_out

    if f.stat().st_size == 0:
        return ret(False), 1
    if f.stat().st_size > max_len:
        return ret(False), 2

    try:
        header = pe.PE(f.as_posix()).FILE_HEADER
    except pe.PEFormatError:
        return ret(False), 3

    if header.IMAGE_FILE_16BIT_MACHINE and not _16_bit:
        return ret(False), 4
    if header.IMAGE_FILE_32BIT_MACHINE and not _32_bit:
        return ret(False), 5
    if not header.IMAGE_FILE_16BIT_MACHINE and not header.IMAGE_FILE_32BIT_MACHINE and not _64_bit:
        return ret(False), 6

    return ret(True), 0


def unpack(f: Path, dest_path: Path) -> Path:
    f_out = dest_path / f.name
    command = [cfg.UPX, "-d", f"{f.as_posix()}", "-o", f"{f_out.as_posix()}"]
    result = subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
    if result.returncode == 1:
        f.rename(f_out)
    elif result.returncode == 2:
        f.rename(f_out)
    else:
        f.unlink()
    return f_out


def extract(f: Path, dest_path: Path) -> Path:
    f_out = (dest_path / f.name).with_suffix(".exe")
    with open(f, "rb") as handle:
        extracted = zlib.decompress(handle.read())
    with open(f_out, "wb") as handle:
        handle.write(extracted)
    f.unlink()
    return f_out


def download_sorel(dest_path: Path, n_files: int) -> tp.Generator[Path, None, None]:
    print("Downloading SOREL...", flush=True)

    yielded = set()
    command = f"{cfg.AWS} s3 cp {cfg.BUCKET} {dest_path} --recursive --no-sign-request --quiet"
    pro = subprocess.Popen(command, shell=True)
    while pro.poll() is None:
        for p in dest_path.iterdir():
            if p.suffix == "":
                p.rename(p.with_suffix(".zlib"))
            if p.name not in yielded and p.suffix == ".zlib":
                yielded.add(p.name)
                yield p
        if len(yielded) >= n_files:
            break

    time.sleep(WAIT)
    os.system("killall aws")

    for p in dest_path.iterdir():
        if p.suffix == "":
            p.rename(p.with_suffix(".zlib"))
            yield p
        else:
            p.unlink()


def download_windows(dest_path: Path, n_files: int) -> tp.Generator[Path, None, None]:
    print("Downloading Windows...", flush=True)

    yielded = set()
    dest_path.mkdir(exist_ok=True)
    command = f"scp {cfg.WINDOWS_BUCKET} {dest_path.as_posix()}/"
    pro = subprocess.Popen(command, shell=True, preexec_fn=os.setsid, stdout=subprocess.PIPE)
    while pro.poll() is None:
        for p in dest_path.iterdir():
            if p.name not in yielded and p.suffix == ".exe":
                yielded.add(p.name)
                yield p
        if len(yielded) >= n_files:
            break

    time.sleep(WAIT)
    os.killpg(os.getpgid(pro.pid), signal.SIGTERM)

    for p in dest_path.iterdir():
        if p.suffix == ".exe":
            yield p


def process_binary(om: OutputManager, f: Path, max_len: int, _16_bit: bool, _32_bit: bool, _64_bit: bool) -> None:
    if f.suffix == ".zlib":
        f = extract(f, om.extract)
        if not f.exists():
            return 1
    f = unpack(f, om.unpack)
    if not f.exists():
        return 2
    f, _ = filter_(f, om.filter, max_len, _16_bit, _32_bit, _64_bit)
    if not f.exists():
        return 3
    files = disassemble(f, om.disassemble)
    for f in files:
        f = pre_normalize(f, om.pre_normalized)
    return 0


def has_file(path: Path, suffix: str = "") -> bool:
    i = 0
    for i, f in enumerate(path.iterdir()):
        if f.suffix == suffix:
            return True
    return i == 0


def is_responsible(p: Path, responsibility: str) -> bool:
    if responsibility == "ALL":
        return True
    if responsibility == "None":
        return False
    return p.name[0 : len(responsibility)] == responsibility


def no_download_file_iterable(om: OutputManager, responsibility: str) -> tp.Generator[Path, None, None]:

    for p in chain(
        (p for p in om.download_sorel.iterdir() if p.suffix == ".zlib" and is_responsible(p, responsibility)),
        (p for p in om.download_windows.iterdir() if p.suffix == ".exe" and is_responsible(p, responsibility)),
    ):
        yield p


def main(
    n_sorel_files: int = N_FILES,
    n_windows_files: int = N_FILES,
    max_len: int = MAX_LEN,
    _16_bit: bool = _16_BIT,
    _32_bit: bool = _32_BIT,
    _64_bit: bool = _64_BIT,
    no_download: bool = False,
    responsibility: str = "ALL",
) -> None:
    assert _16_bit or _32_bit or _64_bit
    om = OutputManager()
    om.mkdir_prepare_paths(exist_ok=True, parents=True)
    ALREADY_DONE.update(p.stem for p in OutputManager().pre_normalized.iterdir())

    if no_download:
        file_iterable = no_download_file_iterable(om, responsibility)
    else:
        file_iterable = chain(
            download_sorel(om.download_sorel, n_sorel_files), download_windows(om.download_windows, n_windows_files)
        )

    while True:
        for f in tqdm(file_iterable, total=n_sorel_files + n_windows_files):
            if not is_responsible(f, responsibility):
                continue
            if f.stem in ALREADY_DONE:
                continue
            process_binary(om, f, max_len, _16_bit, _32_bit, _64_bit)

        if no_download:
            file_iterable = no_download_file_iterable(om, responsibility)
        else:
            break

    # om.rmdir_prepare_paths(ignore_errors=True)


def debug() -> None:
    main(N_FILES, N_FILES, 1e6, False, True, False)


def cli() -> None:
    parser = ArgumentParser()

    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--n_sorel_files", type=int, default=N_FILES)
    parser.add_argument("--n_windows_files", type=int, default=N_FILES)
    parser.add_argument("--max_len", type=int, default=MAX_LEN)
    parser.add_argument("--_16_bit", action="store_true")
    parser.add_argument("--_32_bit", action="store_true")
    parser.add_argument("--_64_bit", action="store_true")
    parser.add_argument("--no_download", action="store_true")
    parser.add_argument("--responsibility", type=str)

    args = parser.parse_args()

    if args.debug:
        debug()
        return

    main(
        args.n_sorel_files,
        args.n_windows_files,
        args.max_len,
        args._16_bit,
        args._32_bit,
        args._64_bit,
        args.no_download,
        args.responsibility,
    )


if __name__ == "__main__":
    cli()
