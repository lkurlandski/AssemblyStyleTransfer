"""
Acquire data for learning, including downloading and disassembling (Network-intensive).
"""

from argparse import ArgumentParser
from collections.abc import Collection
import csv
from dataclasses import dataclass
from collections import defaultdict
from itertools import chain
import os
import json
from pathlib import Path
import pefile as pe
from pprint import pprint
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


N_FILES = 100
MAX_LEN = 1e6
_16_BIT = False
_32_BIT = True
_64_BIT = False
WAIT = 5

WARNS = [("Functions with same addresses detected", False), ("Function addresses improperly parsed", False)]


def pre_normalize(f: Path, dest_path: Path) -> Path:
    normalizer = get_pre_normalizer()
    d_out = dest_path / f.parent.name
    d_out.mkdir(exist_ok=True)
    f_out = d_out / f.name
    with open(f) as handle:
        out_str = normalizer.normalize_str(handle.read())
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
) -> Path:
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


def download_sorel(dest_path: Path, n_files: int) -> None:
    print("Downloading SOREL...", flush=True)

    command = f"{cfg.AWS} s3 cp {cfg.BUCKET} {dest_path} --recursive --no-sign-request --quiet"
    pro = subprocess.Popen(command, shell=True)
    with tqdm(total=n_files) as pbar:
        while pro.poll() is None:
            n = sum(1 for _ in dest_path.iterdir())
            if n >= n_files:
                break
            pbar.update(n)

    time.sleep(WAIT)
    os.system("killall aws")

    for f in dest_path.iterdir():
        if f.suffix == "":
            f.rename(f.with_suffix(".zlib"))
        else:
            f.unlink()

    for i, f in enumerate(dest_path.iterdir()):
        if i >= n_files:
            f.unlink()


def download_windows(dest_path: Path, n_files: int) -> None:
    print("Downloading Windows...", flush=True)

    dest_path.mkdir(exist_ok=True)
    command = f"scp {cfg.WINDOWS_BUCKET} {dest_path.as_posix()}/"
    pro = subprocess.Popen(command, shell=True, preexec_fn=os.setsid, stdout=subprocess.PIPE)
    with tqdm(total=n_files) as pbar:
        while pro.poll() is None:
            n = sum(1 for _ in dest_path.iterdir())
            if n >= n_files:
                break
            pbar.update(n)

    time.sleep(WAIT)
    os.killpg(os.getpgid(pro.pid), signal.SIGTERM)

    for i, f in enumerate(dest_path.iterdir()):
        if i >= n_files:
            f.unlink()


def main(
    n_sorel_files: int = N_FILES,
    n_windows_files: int = N_FILES,
    max_len: int = MAX_LEN,
    _16_bit: bool = _16_BIT,
    _32_bit: bool = _32_BIT,
    _64_bit: bool = _64_BIT,
) -> None:
    assert _16_bit or _32_bit or _64_bit
    om = OutputManager()
    om.mkdir_prepare_paths(exist_ok=True, parents=True)

    download_sorel(om.download_sorel, n_sorel_files)
    download_windows(om.download_windows, n_windows_files)

    stats = [0, 0, 0, 0]
    filter_codes = defaultdict(lambda: 0)

    files = chain(om.download_sorel.iterdir(), om.download_windows.iterdir())
    total = n_sorel_files + n_windows_files
    for i, f in tqdm(enumerate(files), total=total):
        if i < n_sorel_files:
            f = extract(f, om.extract)
            if not f.exists():
                continue
        stats[0] += 1
        f = unpack(f, om.unpack)
        if not f.exists():
            continue
        stats[1] += 1
        f, filter_code = filter_(f, om.filter, max_len, _16_bit, _32_bit, _64_bit)
        if not f.exists():
            filter_codes[filter_code] += 1
            continue
        stats[2] += 1
        files = disassemble(f, om.disassemble)
        for f in files:
            f = pre_normalize(f, om.pre_normalized)
        stats[3] += 1

    pprint(stats)
    pprint(filter_codes)
    om.rmdir_prepare_paths(ignore_errors=True)


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

    args = parser.parse_args()

    if args.debug:
        debug()
        return

    main(args.n_sorel_files, args.n_windows_files, args.max_len, args._16_bit, args._32_bit, args._64_bit)


if __name__ == "__main__":
    cli()
