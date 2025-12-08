from dataclasses import dataclass, field
from pathlib import Path

@dataclass(frozen=True)
class Paths:
    # ルートディレクトリ
    root: Path = field(
        default_factory=lambda: Path(__file__).resolve().parent.parent.parent
    )

    # 主要ディレクトリ
    src: Path = field(init=False)
    gen: Path = field(init=False)
    input: Path = field(init=False)
    output: Path = field(init=False)
    utils: Path = field(init=False)

    # 先行研究のディレクトリ
    hpe: Path = field(init=False)
    sapiens: Path = field(init=False)

    def __post_init__(self):
        # root以下
        object.__setattr__(self, "input", self.root / "input")
        object.__setattr__(self, "output", self.root / "output")

        # src 以下
        object.__setattr__(self, "src", self.root / "src")
        object.__setattr__(self, "gen", self.src / "gen")
        object.__setattr__(self, "utils", self.src / "utils")
        object.__setattr__(self, "hpe", self.src / "hpe")
        object.__setattr__(self, "sapiens", self.hpe / "sapiens")

    
    def make_dirs(self):
        dirs_to_create = [
            self.gen,
            self.input,
            self.output,
            self.utils
        ]
        for p in dirs_to_create:
            p.mkdir(parents=True, exist_ok=True)

PATHS = Paths()

if __name__ == "__main__":
    PATHS.make_dirs()