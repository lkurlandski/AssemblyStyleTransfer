"""
Acquire data for learning, including downloading and disassembling (Network-intensive).
"""

from argparse import ArgumentParser
from collections.abc import Collection
import csv
from dataclasses import dataclass
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


N_FILES = 50
MAX_LEN = 1e6
X_86 = True
X_64 = False
WAIT = 5


def pre_normalize(f: Path, dest_path: Path) -> Path:
    normalizer = get_pre_normalizer()
    f_out = dest_path / f.name
    with open(f) as handle:
        out_str = normalizer.normalize_str(f.read())
    with open(f_out, "w") as handle:
        handle.write(out_str)
    return f_out


class PDRParser:

    start_char = "┌"
    end_char = "└"

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
                    start = int(self.extract_address(line), 16)
                    look_for_start = False
                except ValueError:
                    start = None
            if line[0] == self.end_char:  # indicates end of a function
                try:
                    end = int(self.extract_address(line), 16)
                except ValueError:
                    end = None
                func.append(line)
                return start, end, func
            if line[0] == self.start_char:  # indicates start of function
                if add:  # TODO: indicates a double-nested function, so we just start over
                    func = []
                add = True
                look_for_start = True
                start = None
            if add:
                func.append(line)
        if func:
            return start, end, func
        raise StopIteration

    @staticmethod
    def extract_address(line: str):
        return line[2:13]


def disassemble(f: Path, dest_path: Path) -> list[Path]:
    outpath = dest_path / f.stem
    outpath.mkdir()
    parser = PDRParser(f)
    for start, end, func in parser:
        f_out = outpath / f"{start}_{end}.asm"
        if f_out.exists():
            f_out = f_out.with_stem(f_out.stem + "_")  # FIXME: addresses may be incorrect
        with open(f_out, "w") as handle:
            handle.write("\n".join(func))
    return list(outpath.iterdir())


def filter_(
    f: Path,
    dest_path: Path,
    max_len: int = MAX_LEN,
    x_86: bool = X_86,
    x_64: bool = X_64,
) -> Path:
    f_out = dest_path / f.name
    
    keep = True
    if f.stat().st_size == 0:
        keep = keep and False
    if f.stat().st_size > max_len:
        keep = keep and False
    try:
        header = pe.PE(f.as_posix()).FILE_HEADER
        keep = keep and (header.IMAGE_FILE_32BIT_MACHINE == x_86)
        keep = keep and (header.IMAGE_FILE_64BIT_MACHINE == x_64)
    except Exception:
        keep = keep and False

    if keep:
        f.rename(f_out)
    else:
        f.unlink()
    return f_out


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
    subprocess.Popen(command, shell=True)
    with tqdm(total=n_files) as pbar:
        while True:
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
        while True:
            n = sum(1 for _ in dest_path.iterdir())
            if n >= n_files:
                break
            pbar.update(n)

    time.sleep(WAIT)
    os.killpg(os.getpgid(pro.pid), signal.SIGTERM)


def main(
    n_sorel_files: int = N_FILES,
    n_windows_files: int = N_FILES,
    max_len: int = MAX_LEN,
    x_86: bool = X_86,
    x_64: bool = X_64,
) -> None:
    assert x_86 or x_64
    om = OutputManager()
    om.mkdir(exist_ok=True, parents=True)

    download_sorel(om.download_sorel, n_sorel_files)
    download_windows(om.download_windows, n_windows_files)

    for i, f in enumerate(chain(om.download_sorel.iterdir(), om.download_windows.iterdir())):
        if i < n_sorel_files:
            f = extract(f, om.extract)
            if not f.exists():
                continue
        f = unpack(f, om.unpack)
        if not f.exists():
            continue
        f = filter_(f, om.filter, max_len, x_86, x_64)
        if not f.exists():
            continue
        files = disassemble(f, om.disassemble)
        for f in files:
            f = pre_normalize(f)


def debug() -> None:
    ...


def cli() -> None:
    parser = ArgumentParser()

    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--n_sorel_files", type=int, default=N_FILES)
    parser.add_argument("--n_windows_files", type=int, default=N_FILES)
    parser.add_argument("--max_len", type=int, default=MAX_LEN)
    parser.add_argument("--x_86", action="store_true")
    parser.add_argument("--x_64", action="store_true")

    args = parser.parse_args()

    if args.debug:
        debug()
        return

    main(
        args.n_sorel_files,
        args.n_windows_files,
        args.max_len,
        args.x_86,
        args.x_64,
    )


if __name__ == "__main__":
    cli()
