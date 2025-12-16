from dataclasses import dataclass
import numpy as np


@dataclass
class Point3D:
    x: float
    y: float
    z: float

    def to_array(self) -> np.ndarray:
        return np.array([self.x, self.y, self.z], dtype=np.float32)


class VectorMath:
    @staticmethod
    def normalize(v: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(v)
        if norm < 1e-10:
            return v
        return v / norm

    @staticmethod
    def cross_product_normalized(v1: np.ndarray, v2: np.ndarray) -> np.ndarray:
        return VectorMath.normalize(np.cross(v1, v2))

    @staticmethod
    def angle_between(v1: np.ndarray, v2: np.ndarray) -> float:
        dot_val = np.clip(np.dot(v1, v2), -1.0, 1.0)
        return np.degrees(np.arccos(dot_val))

    @staticmethod
    def rotate_around_axis(v: np.ndarray, axis: np.ndarray, angle_deg: float) -> np.ndarray:
        rad = np.radians(angle_deg)
        cos_theta = np.cos(rad)
        sin_theta = np.sin(rad)
        cross_term = np.cross(axis, v)
        return v * cos_theta + cross_term * sin_theta
