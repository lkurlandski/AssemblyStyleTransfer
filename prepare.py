"""
Acquire data for learning, including downloading and disassembling (Network-intensive).
"""

from argparse import ArgumentParser
from collections.abc import Collection
import csv
from dataclasses import dataclass
import os
import json
from pathlib import Path
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
from utils import (
    disasm,
    instruction_as_str,
    get_text_section_bounds,
    maybe_remove,
    read_file,
    verify_text_section_bounds,
    OutputManager,
)


def disassemble(files: Collection[Path], dest_path: Path, bounds_file: Path, address: bool) -> None:
    print("Disassembling...", flush=True)

    # name_file_map = {f.name: f for f in files}

    # with open(bounds_file, encoding="utf-8") as handle:
    #     reader = csv.reader(handle)
    #     next(reader)
    #     for row in tqdm(reader, total=len(name_file_map)):
    #         if not row:
    #             break
    #         f, l, u, _ = row
    #         f = name_file_map[Path(f).name]
    #         l = int(l)
    #         u = int(u)
            
    #         r2 = r2pipe.open(f.as_posix())
    #         r2.cmd("aa")
    #         instructions = r2.cmd("pd $s")
    #         with open((dest_path / f.name).with_suffix(".asm"), "w", encoding="utf-8") as handle_:
    #             handle_.write(instructions)

    for f in files:
        r2 = r2pipe.open(f.as_posix())
        r2.cmd("aa")
        instructions = r2.cmd("pd $s")
        with open((dest_path / f.name).with_suffix(".asm"), "w", encoding="utf-8") as handle_:
            handle_.write(instructions)


def parse(files: Collection[Path], dest_path: Path, bounds_file: Path, posix: bool, remove: bool) -> None:
    print("Parsing...", flush=True)

    with open(bounds_file, mode="w", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["file", "lower", "upper", "size"])
        for f in tqdm(files):
            if (o := get_text_section_bounds(f, errors="quiet")) is None:
                maybe_remove(f, remove)
                continue
            l, u = o
            if verify_text_section_bounds(f, l, u, False) != 0:
                maybe_remove(f, remove)
                continue
            f_out = dest_path / f.name
            shutil.copy(f, f_out)
            name = f_out.as_posix() if posix else f_out.name
            writer.writerow([name, l, u, f_out.stat().st_size])
            maybe_remove(f, remove)
        handle.write("\n")


def filter_(files: Collection[Path], dest_path: Path, max_len: int, remove: bool) -> None:
    print("Filtering...", flush=True)

    def fn(p: Path) -> bool:
        if p.stat().st_size == 0:
            return False
        if p.stat().st_size > max_len:
            return False
        return True

    for f in tqdm(files):
        if not fn(f):
            maybe_remove(f, remove)
            continue
        if remove:
            f.rename(dest_path / f.name)
        else:
            shutil.copy(f, dest_path / f.name)


def unpack(files: Collection[Path], dest_path: Path, remove: bool) -> None:
    print("Unpacking...", flush=True)

    command = [cfg.UPX, "-d", "{PACKED_FILE}", "-o", "{DEST_PATH}"]

    for f in tqdm(files):
        f_out = dest_path / f.name
        command[2] = f.as_posix()
        command[4] = (dest_path / f.name).as_posix()
        result = subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
        if result.returncode == 1:
            shutil.copy(f, f_out)
        elif result.returncode == 2:
            shutil.copy(f, f_out)
        else:
            warnings.warn(result)

        maybe_remove(f, remove)


def extract(files: Collection[Path], dest_path: Path, remove: bool) -> None:
    print("Extracting...", flush=True)

    dest_path.mkdir(exist_ok=True)

    for f in tqdm(files):
        with open(f, "rb") as handle:
            extracted = zlib.decompress(handle.read())
        with open((dest_path / f.name).with_suffix(".exe"), "wb") as handle:
            handle.write(extracted)
        maybe_remove(f, remove)


def download(dest_path: Path, n_files: int) -> None:
    WAIT = 5

    print("Downloading SOREL...", flush=True)

    dest_path.mkdir(exist_ok=True)
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
        elif f.suffix == ".exe":
            pass
        else:
            f.unlink()

    for i, f in enumerate(dest_path.iterdir()):
        if i >= n_files:
            f.unlink()


def download_windows(dest_path: Path, n_files: int) -> None:
    WAIT = 5

    print("Downloading Windows...", flush=True)

    dest_path.mkdir(exist_ok=True)
    command = f"scp {cfg.WINDOWS_BUCKET} {dest_path.as_posix()}/"
    print(command)
    #sys.exit()
    pro = subprocess.Popen(command, shell=True, preexec_fn=os.setsid, stdout=subprocess.PIPE)
    with tqdm(total=n_files) as pbar:
        while True:
            n = sum(1 for _ in dest_path.iterdir())
            if n >= n_files * 2:
                break
            pbar.update(n)

    time.sleep(WAIT)
    os.killpg(os.getpgid(pro.pid), signal.SIGTERM)


@dataclass
class ParamArgs:
    n_files: int = 10
    max_len: int = 16e6
    posix: bool = False
    address: bool = True


@dataclass
class ActionArgs:
    download: bool = False
    download_windows: bool = False
    extract: bool = False
    unpack: bool = False
    filter: bool = False
    parse: bool = False
    disassemble: bool = False


@dataclass
class CleanArgs:
    download: bool = False
    extract: bool = False
    unpack: bool = False
    filter: bool = False
    parse: bool = False
    disassemble: bool = False


@dataclass
class RemoveArgs:
    download: bool = False
    extract: bool = False
    unpack: bool = False
    filter: bool = False


def main(
    params: ParamArgs,
    paths: OutputManager,
    actions: ActionArgs,
    removes: RemoveArgs,
    clean: CleanArgs,
) -> None:
    # TODO: refactor into OutputManager
    if clean.download:
        shutil.rmtree(paths.download, ignore_errors=True)
    if clean.extract:
        shutil.rmtree(paths.extract, ignore_errors=True)
    if clean.unpack:
        shutil.rmtree(paths.unpack, ignore_errors=True)
    if clean.filter:
        shutil.rmtree(paths.filter, ignore_errors=True)
    if clean.parse:
        paths.bounds_file.unlink(missing_ok=True)
        shutil.rmtree(paths.parse, ignore_errors=True)
    if clean.disassemble:
        shutil.rmtree(paths.disassemble, ignore_errors=True)

    paths.mkdir(exist_ok=True, parents=True)

    if actions.download:
        paths.download.mkdir(exist_ok=True)
        download(paths.download, params.n_files)
    if actions.extract:
        paths.extract.mkdir(exist_ok=True)
        extract(list(paths.download.iterdir()), paths.extract, removes.download)
        if removes.download:
            paths.download.rmdir()
    if actions.download_windows:
        paths.extract.mkdir(exist_ok=True)
        download_windows(paths.extract, params.n_files)

    if actions.unpack:
        paths.unpack.mkdir(exist_ok=True)
        unpack(list(paths.extract.iterdir()), paths.unpack, removes.extract)
        if removes.extract:
            paths.extract.rmdir()

    if actions.filter:
        paths.filter.mkdir(exist_ok=True)
        filter_(list(paths.unpack.iterdir()), paths.filter, params.max_len, removes.unpack)
        if removes.unpack:
            paths.unpack.rmdir()

    if actions.parse:
        paths.parse.mkdir(exist_ok=True)
        parse(list(paths.filter.iterdir()), paths.parse, paths.bounds_file, params.posix, removes.filter)
        if removes.filter:
            paths.filter.rmdir()

    if actions.disassemble:
        paths.disassemble.mkdir(exist_ok=True)
        disassemble(list(paths.parse.iterdir()), paths.disassemble, paths.bounds_file, params.address)


def debug() -> None:
    main(
        params=ParamArgs(),
        paths=OutputManager(),
        actions=ActionArgs(disassemble=True),
        removes=RemoveArgs(),
        clean=CleanArgs(),
    )


def cli() -> None:
    parser = ArgumentParser()

    parser.add_argument("--debug", action="store_true", help="DEBUG")

    parser.add_argument("--all", action="store_true", help="ACTION")
    parser.add_argument("--download", action="store_true", help="ACTION")
    parser.add_argument("--download_windows", action="store_true", help="ACTION")
    parser.add_argument("--extract", action="store_true", help="ACTION")
    parser.add_argument("--unpack", action="store_true", help="ACTION")
    parser.add_argument("--filter", action="store_true", help="ACTION")
    parser.add_argument("--parse", action="store_true", help="ACTION")
    parser.add_argument("--disassemble", action="store_true", help="ACTION")

    parser.add_argument("--n_files", type=int, default=ParamArgs.n_files, help="PARAM")
    parser.add_argument("--max_len", type=int, default=ParamArgs.max_len, help="PARAM")
    parser.add_argument("--posix", action="store_true", help="PARAM")
    parser.add_argument("--no_address", action="store_true", help="PARAM")

    parser.add_argument("--clean_all", action="store_true", help="CLEAN-BEFORE")
    parser.add_argument("--clean_download", action="store_true", help="CLEAN-BEFORE")
    parser.add_argument("--clean_extract", action="store_true", help="CLEAN-BEFORE")
    parser.add_argument("--clean_unpack", action="store_true", help="CLEAN-BEFORE")
    parser.add_argument("--clean_filter", action="store_true", help="CLEAN-BEFORE")
    parser.add_argument("--clean_parse", action="store_true", help="CLEAN-BEFORE")
    parser.add_argument("--clean_disassemble", action="store_true", help="CLEAN-BEFORE")

    parser.add_argument("--remove_all", action="store_true", help="REMOVE-AFTER")
    parser.add_argument("--remove_download", action="store_true", help="REMOVE-AFTER")
    parser.add_argument("--remove_extract", action="store_true", help="REMOVE-AFTER")
    parser.add_argument("--remove_unpack", action="store_true", help="REMOVE-AFTER")
    parser.add_argument("--remove_filter", action="store_true", help="REMOVE-AFTER")

    args = parser.parse_args()

    if args.debug:
        debug()
        return

    main(
        params=ParamArgs(
            args.n_files,
            args.max_len,
            args.posix,
            not args.no_address,
        ),
        paths=OutputManager(),
        actions=ActionArgs(
            args.download or args.all,
            args.download_windows or args.all,
            args.extract or args.all,
            args.unpack or args.all,
            args.filter or args.all,
            args.parse or args.all,
            args.disassemble or args.all,
        ),
        removes=RemoveArgs(
            args.remove_download or args.remove_all,
            args.remove_extract or args.remove_all,
            args.remove_unpack or args.remove_all,
            args.remove_filter or args.remove_all,
        ),
        clean=CleanArgs(
            args.clean_download or args.clean_all,
            args.clean_extract or args.clean_all,
            args.clean_unpack or args.clean_all,
            args.clean_filter or args.clean_all,
            args.clean_parse or args.clean_all,
            args.clean_disassemble or args.clean_all,
        ),
    )


if __name__ == "__main__":
    cli()
