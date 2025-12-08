from dataclasses import dataclass, field
from pathlib import Path


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
    def raw_video(self, camera: str, person: str, condition: str, ext="mp4"):
        """
        data/raw/video/fisheye/person00/light_on.mp4
        """
        return self.base.raw / "video" / camera / person / f"{condition}.{ext}"

    def raw_image(self, camera: str, person: str, condition: str, ext="jpg"):
        """
        data/raw/image/fisheye/person00/light_on.jpg
        """
        return self.base.raw / "image" / camera / person / f"{condition}.{ext}"

    # ----------------------
    # Interim data
    # ----------------------
    def frames_dir(self, camera: str, person: str, condition: str, surface: str):
        """
        前処理後の frame 出力先:
        data/interim/frames/fisheye/person00/light_on/left
        data/interim/frames/fisheye/person00/light_on/right
        """
        return self.base.interim / "frames" / camera / person / condition / surface

    def calibration_interim(self):
        """
        data/interim/calibration/
        """
        return self.base.interim / "calibration"

    # ----------------------
    # Processed data
    # ----------------------
    def hpe_json_dir(self, model: str, camera: str, person: str, condition: str):
        """
        data/processed/hpe_json/sapiens/fisheye/person00/light_on/
        """
        return self.base.processed / "hpe_json" / model / camera / person / condition

    def hpe_vis_dir(self, model: str, camera: str, person: str, condition: str):
        """
        data/processed/hpe_vis/sapiens/fisheye/person00/light_on/
        """
        return self.base.processed / "hpe_vis" / model / camera / person / condition

    def hpe_normalized_dir(self, model: str, camera: str, person: str, condition: str):
        """
        data/processed/hpe_normalized/sapiens/fisheye/person00/light_on/
        """
        return self.base.processed / "hpe_normalized" / model / camera / person / condition

    def skeleton_dir(self, model: str, camera: str, person: str, condition: str):
        """
        data/processed/gcn_skeletons/sapiens/fisheye/person00/light_on/
        """
        return self.base.processed / "gcn_skeletons" / model / camera / person / condition

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


if __name__ == "__main__":
    PATHS.make_dirs()
