import numpy as np
from abc import ABC, abstractmethod


class IPointCloudWriter(ABC):
    @abstractmethod
    def write(self, filepath: str, points: np.ndarray, colors: np.ndarray):
        ...


class BinaryPLYWriter(IPointCloudWriter):
    def write(self, filepath: str, points: np.ndarray, colors: np.ndarray):
        points, colors = points.astype(np.float32), colors.astype(np.uint8)
        n = len(points)
        header = (
            "ply\nformat binary_little_endian 1.0\n"
            f"element vertex {n}\n"
            "property float x\nproperty float y\nproperty float z\n"
            "property uchar red\nproperty uchar green\nproperty uchar blue\n"
            "end_header\n"
        )
        with open(filepath, "wb") as f:
            f.write(header.encode("ascii"))
            dt = [
                ("x", "f4"),
                ("y", "f4"),
                ("z", "f4"),
                ("r", "u1"),
                ("g", "u1"),
                ("b", "u1"),
            ]
            v = np.empty(n, dtype=dt)
            v["x"], v["y"], v["z"] = points[:, 0], points[:, 1], points[:, 2]
            v["r"], v["g"], v["b"] = colors[:, 0], colors[:, 1], colors[:, 2]
            f.write(v.tobytes())


def load_ply_binary_simple(filepath: str):
    with open(filepath, "rb") as f:
        while f.readline().decode("ascii").strip() != "end_header":
            pass
        dt = [
            ("x", "f4"),
            ("y", "f4"),
            ("z", "f4"),
            ("r", "u1"),
            ("g", "u1"),
            ("b", "u1"),
        ]
        data = np.frombuffer(f.read(), dtype=dt)
        return (
            np.column_stack((data["x"], data["y"], data["z"])),
            np.column_stack((data["r"], data["g"], data["b"])),
        )
