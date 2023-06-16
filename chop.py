"""Chop snippets of assembly code into malicious and benign classes.
"""

from argparse import ArgumentParser
from collections.abc import Collection
from dataclasses import dataclass
import json
from pathlib import Path
import typing as tp
import warnings

from utils import OutputManager


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


def main(paths: OutputManager, ben_threshold: float, mal_threshold: float) -> None:
    paths.snippets.mkdir(exist_ok=True)
    paths.snippets_mal.mkdir(exist_ok=True)
    paths.snippets_ben.mkdir(exist_ok=True)
    chop_snippets(
        list(paths.disassemble.iterdir()),
        paths.snippets_mal,
        paths.snippets_ben,
        paths.summary_file,
        ben_threshold,
        mal_threshold,
    )


def debug() -> None:
    pass


def cli() -> None:
    parser = ArgumentParser()
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--mal_threshold", type=float, default=0.5)
    parser.add_argument("--ben_threshold", type=float, default=-0.5)
    args = parser.parse_args()
    main(OutputManager(), args.ben_threshold, args.mal_threshold)


if __name__ == "__main__":
    cli()
