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
import subprocess
import time
import typing as tp
import warnings
import zlib

import capstone
from tqdm import tqdm

import cfg
from utils import (
    disasm,
    instruction_as_str,
    get_text_section_bounds,
    maybe_remove,
    read_file,
    verify_text_section_bounds,
)


def chop_snippets(
    files: Collection[Path],
    mal_path: Path,
    ben_path: Path,
    summary_file: Path,
    ben_threshold: float,
    mal_threshold: float,
) -> None:
    print("Chopping...", flush=True)

    def get_outpath(f: Path, a: float, i: int) -> tp.Optional[Path]:
        if a < ben_threshold:
            return (ben_path / f"{f.stem}_{i}").with_suffix(".asm")
        if a > mal_threshold:
            return (mal_path / f"{f.stem}_{i}").with_suffix(".asm")
        return None

    def write_snippet(outfile: Path, snippet: list[str]) -> None:
        with open(outfile, "w", encoding="utf-8") as handle_out:
            handle_out.write("\n".join(snippet))

    with open(summary_file, encoding="utf-8") as handle:
        summary = {Path(f).name: v for f, v in json.load(handle).items()}

    for f in files:
        if (key := f.with_suffix(".exe").name) not in summary:
            warnings.warn(f"Found disassembly for {f.name}, but have no attributions for it.")
            continue
        offsets, attribs = list(zip(*summary[key]))
        i, snippet = 0, []

        with open(f, encoding="utf-8") as handle_in:
            for line in handle_in:
                if line == "\n":
                    break
                offset, instruction = line.rstrip().split("\t")
                snippet.append(f"{offset}\t{instruction}")

                if i < len(offsets) - 1 and int(offset, 16) >= offsets[i + 1]:
                    outfile = get_outpath(f, attribs[i], i)
                    if outfile is not None:
                        write_snippet(outfile, snippet)
                    i += 1
                    snippet = []

            if snippet:
                outfile = get_outpath(f, attribs[i], i)
                if outfile is not None:
                    write_snippet(outfile, snippet)


def disassemble(files: Collection[Path], dest_path: Path, bounds_file: Path, address: bool, remove: bool) -> None:
    print("Disassembling...", flush=True)

    def format_fn(ins):
        return instruction_as_str(ins, address=address)

    md = capstone.Cs(cfg.ARCH, cfg.MODE)

    name_file_map = {f.name: f for f in files}

    with open(bounds_file, encoding="utf-8") as handle:
        reader = csv.reader(handle)
        next(reader)
        for row in tqdm(reader, total=len(name_file_map)):
            if not row:
                break
            f, l, u, _ = row
            f = name_file_map[Path(f).name]
            l = int(l)
            u = int(u)

            code = read_file(f, l, u)
            instructions = disasm(md, code, format_fn, start=l)

            with open((dest_path / f.name).with_suffix(".asm"), "w", encoding="utf-8") as handle_:
                handle_.write("\n".join(instructions) + "\n")

            maybe_remove(f, remove)

    if remove:
        for f in files:
            f.unlink()


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

    print("Downloading...", flush=True)

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


@dataclass
class ParamArgs:
    n_files: int = 10
    max_len: int = 16e6
    posix: bool = False
    mal_threshold: float = 0.5
    ben_threshold: float = -0.5
    address: bool = True


@dataclass
class PathArgs:
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

    def __post_init__(self) -> None:
        self.download = self.root / self.download
        self.extract = self.root / self.extract
        self.unpack = self.root / self.unpack
        self.filter = self.root / self.filter
        self.parse = self.root / self.parse
        self.disassemble = self.root / self.disassemble
        self.snippets = self.root / self.snippets
        self.snippets_mal = self.snippets / "mal"
        self.snippets_ben = self.snippets / "ben"
        self.bounds_file = self.root / self.bounds_file
        self.summary_file = self.root / self.summary_file


@dataclass
class ActionArgs:
    download: bool = False
    extract: bool = False
    unpack: bool = False
    filter: bool = False
    parse: bool = False
    disassemble: bool = False
    chop: bool = False


@dataclass
class RemoveArgs:
    download: bool = False
    extract: bool = False
    unpack: bool = False
    filter: bool = False
    parse: bool = False


@dataclass
class CleanArgs:
    download: bool = False
    extract: bool = False
    unpack: bool = False
    filter: bool = False
    parse: bool = False
    disassemble: bool = False
    chop: bool = False


def main(
    params: ParamArgs,
    paths: PathArgs,
    actions: ActionArgs,
    removes: RemoveArgs,
    clean: CleanArgs,
) -> None:
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
    if clean.chop:
        shutil.rmtree(paths.snippets, ignore_errors=True)

    paths.root.mkdir(exist_ok=True, parents=True)

    if actions.download:
        paths.download.mkdir(exist_ok=True)
        download(paths.download, params.n_files)

    if actions.extract:
        paths.extract.mkdir(exist_ok=True)
        extract(list(paths.download.iterdir()), paths.extract, removes.download)
        if removes.download:
            paths.download.rmdir()

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
        disassemble(list(paths.parse.iterdir()), paths.disassemble, paths.bounds_file, params.address, removes.parse)

    if actions.chop:
        paths.snippets.mkdir(exist_ok=True)
        paths.snippets_mal.mkdir(exist_ok=True)
        paths.snippets_ben.mkdir(exist_ok=True)
        chop_snippets(
            list(paths.disassemble.iterdir()),
            paths.snippets_mal,
            paths.snippets_ben,
            paths.summary_file,
            params.ben_threshold,
            params.mal_threshold,
        )


if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument("--root", type=Path, help="Path")

    parser.add_argument("--download", action="store_true", help="ACTION")
    parser.add_argument("--extract", action="store_true", help="ACTION")
    parser.add_argument("--unpack", action="store_true", help="ACTION")
    parser.add_argument("--filter", action="store_true", help="ACTION")
    parser.add_argument("--parse", action="store_true", help="ACTION")
    parser.add_argument("--explain", action="store_true", help="ACTION")
    parser.add_argument("--disassemble", action="store_true", help="ACTION")
    parser.add_argument("--chop", action="store_true", help="ACTION")

    parser.add_argument("--n_files", type=int, default=ParamArgs.n_files, help="PARAM")
    parser.add_argument("--max_len", type=int, default=ParamArgs.max_len, help="PARAM")
    parser.add_argument("--posix", action="store_true", help="PARAM")
    parser.add_argument("--mal_threshold", type=float, default=ParamArgs.mal_threshold, help="PARAM")
    parser.add_argument("--ben_threshold", type=float, default=ParamArgs.ben_threshold, help="PARAM")
    parser.add_argument("--no_address", action="store_true", help="PARAM")

    parser.add_argument("--remove_download", action="store_true", help="REMOVE-AFTER")
    parser.add_argument("--remove_extract", action="store_true", help="REMOVE-AFTER")
    parser.add_argument("--remove_unpack", action="store_true", help="REMOVE-AFTER")
    parser.add_argument("--remove_filter", action="store_true", help="REMOVE-AFTER")
    parser.add_argument("--remove_parse", action="store_true", help="REMOVE-AFTER")

    parser.add_argument("--clean_download", action="store_true", help="CLEAN-BEFORE")
    parser.add_argument("--clean_extract", action="store_true", help="CLEAN-BEFORE")
    parser.add_argument("--clean_unpack", action="store_true", help="CLEAN-BEFORE")
    parser.add_argument("--clean_filter", action="store_true", help="CLEAN-BEFORE")
    parser.add_argument("--clean_parse", action="store_true", help="CLEAN-BEFORE")
    parser.add_argument("--clean_disassemble", action="store_true", help="CLEAN-BEFORE")
    parser.add_argument("--clean_chop", action="store_true", help="CLEAN-BEFORE")

    args = parser.parse_args()

    main(
        params=ParamArgs(
            args.n_files,
            args.max_len,
            args.posix,
            args.mal_threshold,
            args.ben_threshold,
            not args.no_address,
        ),
        paths=PathArgs(
            args.root,
        ),
        actions=ActionArgs(
            args.download,
            args.extract,
            args.unpack,
            args.filter,
            args.parse,
            args.disassemble,
            args.chop,
        ),
        removes=RemoveArgs(
            args.remove_download,
            args.remove_extract,
            args.remove_unpack,
            args.remove_filter,
            args.remove_parse,
        ),
        clean=CleanArgs(
            args.clean_download,
            args.clean_extract,
            args.clean_unpack,
            args.clean_filter,
            args.clean_parse,
            args.clean_disassemble,
            args.clean_chop,
        ),
    )
