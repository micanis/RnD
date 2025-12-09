from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Literal

import questionary


# ------------------------------------------------------------
# Base: プロジェクトが持つ「静的なルートディレクトリ」のみを管理
# ------------------------------------------------------------
@dataclass(frozen=True)
class BasePaths:
    root: Path = field(default_factory=lambda: Path(__file__).resolve().parents[2])

    data: Path = field(init=False)
    raw: Path = field(init=False)
    interim: Path = field(init=False)
    processed: Path = field(init=False)

    experiments: Path = field(init=False)
    models: Path = field(init=False)
    src: Path = field(init=False)
    docs: Path = field(init=False)
    notebooks: Path = field(init=False)
    tmp: Path = field(init=False)

    def __post_init__(self):
        object.__setattr__(self, "data", self.root / "data")
        object.__setattr__(self, "raw", self.data / "raw")
        object.__setattr__(self, "interim", self.data / "interim")
        object.__setattr__(self, "processed", self.data / "processed")

        object.__setattr__(self, "experiments", self.root / "experiments")
        object.__setattr__(self, "models", self.root / "models")
        object.__setattr__(self, "src", self.root / "src")
        object.__setattr__(self, "docs", self.root / "docs")
        object.__setattr__(self, "notebooks", self.root / "notebooks")
        object.__setattr__(self, "tmp", self.root / "tmp")

    def make_dirs(self):
        """プロジェクト作成時に最低限のフォルダを初期化"""
        dirs = [
            self.data, self.raw, self.interim, self.processed,
            self.experiments, self.models, self.src,
            self.docs, self.notebooks, self.tmp
        ]
        for d in dirs:
            d.mkdir(parents=True, exist_ok=True)


# ------------------------------------------------------------
# PathResolver: person / condition / camera に応じて動的に解決
# ------------------------------------------------------------
class PathResolver:
    def __init__(self, base: BasePaths):
        self.base = base

    # ----------------------
    # Raw data
    # ----------------------
    def raw_video(self, camera: str, subject: str, condition: str, ext="mp4"):
        """
        data/raw/video/fisheye/person00/light_on.mp4
        """
        return self.base.raw / "video" / camera / subject / f"{condition}.{ext}"

    def raw_image(self, camera: str, subject: str, condition: str, ext="jpg"):
        """
        data/raw/image/fisheye/person00/light_on.jpg
        """
        return self.base.raw / "image" / camera / subject / f"{condition}.{ext}"

    # ----------------------
    # Interim data
    # ----------------------
    def frames_dir(self, camera: str, subject: str, condition: str, surface: str):
        """
        前処理後の frame 出力先:
        data/interim/frames/fisheye/person00/light_on/left
        data/interim/frames/fisheye/person00/light_on/right
        """
        return self.base.interim / "frames" / camera / subject / condition / surface

    def calibration_interim(self):
        """
        data/interim/calibration/
        """
        return self.base.interim / "calibration"

    # ----------------------
    # Processed data
    # ----------------------
    def hpe_json_dir(self, model: str, camera: str, subject: str, condition: str):
        """
        data/processed/hpe_json/sapiens/fisheye/person00/light_on/
        """
        return self.base.processed / "hpe_json" / model / camera / subject / condition

    def hpe_vis_dir(self, model: str, camera: str, subject: str, condition: str):
        """
        data/processed/hpe_vis/sapiens/fisheye/person00/light_on/
        """
        return self.base.processed / "hpe_vis" / model / camera / subject / condition

    def hpe_normalized_dir(self, model: str, camera: str, subject: str, condition: str):
        """
        data/processed/hpe_normalized/sapiens/fisheye/person00/light_on/
        """
        return self.base.processed / "hpe_normalized" / model / camera / subject / condition

    def skeleton_dir(self, model: str, camera: str, subject: str, condition: str):
        """
        data/processed/gcn_skeletons/sapiens/fisheye/person00/light_on/
        """
        return self.base.processed / "gcn_skeletons" / model / camera / subject / condition

    # ----------------------
    # Experiments
    # ----------------------
    def experiment(self, exp_name: str):
        """
        experiments/exp001_sapiens_pose_estimation/
        """
        return self.base.experiments / exp_name

    def experiment_results(self, exp_name: str):
        """
        experiments/expXXX/results/
        """
        return self.experiment(exp_name) / "results"

    # ----------------------
    # Utility
    # ----------------------
    @staticmethod
    def ensure(*paths: Path):
        """必要なフォルダをまとめて作成"""
        for p in paths:
            p.mkdir(parents=True, exist_ok=True)


# ------------------------------------------------------------
# Global instance
# ------------------------------------------------------------
PATHS = BasePaths()
RESOLVE = PathResolver(PATHS)


def _normalize_extensions(extensions: Iterable[str] | None) -> set[str]:
    if not extensions:
        return set()
    return {
        ext if ext.startswith(".") else f".{ext}"
        for ext in (ext.lower() for ext in extensions)
    }


def ask_path(
    message: str,
    base_dir: Path | None = None,
    *,
    kind: Literal["dir", "file"] = "dir",
    extensions: Iterable[str] | None = None,
    choices: Iterable[Path] | None = None,
    allow_manual: bool = True,
    create: bool = False,
) -> Path | None:
    """
    questionaryでファイル/フォルダパスを選択するヘルパー。

    - choices が与えられていればそれを候補とする。
    - base_dir が与えられればその直下をスキャンして候補を作る。
    - allow_manual=True なら任意のパス入力も受け付ける。
    - kind == \"dir\" かつ create=True なら選択後にフォルダを自動作成する。
    """
    base_dir = base_dir or PATHS.root
    manual_token = "__manual__"
    normalized_exts = _normalize_extensions(extensions)

    if create and kind == "dir":
        base_dir.mkdir(parents=True, exist_ok=True)

    if choices is not None:
        candidates = sorted(Path(c) for c in choices)
    elif base_dir.exists():
        if kind == "dir":
            candidates = sorted(p for p in base_dir.iterdir() if p.is_dir())
        else:
            candidates = sorted(
                p
                for p in base_dir.iterdir()
                if p.is_file() and (not normalized_exts or p.suffix.lower() in normalized_exts)
            )
    else:
        candidates = []

    question_choices = [questionary.Choice(str(p), value=p) for p in candidates]
    if allow_manual:
        question_choices.append(questionary.Choice("別のパスを指定する", value=manual_token))

    if not question_choices:
        return None

    selected = questionary.select(message, choices=question_choices).ask()
    if selected is None:
        return None

    if selected == manual_token:
        manual_input = questionary.path(
            "パスを入力してください:",
            default=str(base_dir),
            only_directories=kind == "dir",
        ).ask()
        if not manual_input:
            return None
        selected_path = Path(manual_input).expanduser()
    else:
        selected_path = Path(selected)

    if not selected_path.exists():
        if kind == "dir" and create:
            selected_path.mkdir(parents=True, exist_ok=True)
        else:
            raise FileNotFoundError(f"指定されたパスが存在しません: {selected_path}")
    return selected_path


if __name__ == "__main__":
    PATHS.make_dirs()
