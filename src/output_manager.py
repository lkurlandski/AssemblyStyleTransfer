"""

"""

from pathlib import Path


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
        self.merged = self.data / "merged"

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
            self.merged,
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
