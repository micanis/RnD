import numpy as np
from abc import ABC, abstractmethod
from typing import Tuple


class ICameraModel(ABC):
    @abstractmethod
    def pixel_to_vector(self, u: float, v: float) -> np.ndarray:
        ...

    @abstractmethod
    def vector_to_pixel(self, vec: np.ndarray) -> Tuple[float, float]:
        ...

    @abstractmethod
    def generate_background_cloud(self, img: np.ndarray, step: int) -> Tuple[np.ndarray, np.ndarray]:
        ...

    @property
    @abstractmethod
    def resolution(self) -> Tuple[int, int]:
        ...


class FisheyeCamera(ICameraModel):
    def __init__(self, width: int, height: int, fov_deg: float = 180.0):
        self.W = width
        self.H = height
        self.cx = width / 2.0
        self.cy = height / 2.0
        self.R = min(width, height) / 2.0
        fov_rad = np.radians(fov_deg)
        self.f = self.R / (fov_rad / 2.0)

    @property
    def resolution(self) -> Tuple[int, int]:
        return self.W, self.H

    def pixel_to_vector(self, u: float, v: float) -> np.ndarray:
        dx, dy = u - self.cx, v - self.cy
        r = np.sqrt(dx**2 + dy**2)
        phi = np.arctan2(dy, dx)
        theta = r / self.f
        return np.array(
            [np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)],
            dtype=np.float32,
        )

    def vector_to_pixel(self, vec: np.ndarray) -> Tuple[float, float]:
        x, y, z = vec
        theta = np.arccos(np.clip(z, -1.0, 1.0))
        r = self.f * theta
        phi = np.arctan2(y, x)
        return self.cx + r * np.cos(phi), self.cy + r * np.sin(phi)

    def generate_background_cloud(self, img: np.ndarray, step: int = 4) -> Tuple[np.ndarray, np.ndarray]:
        y, x = np.mgrid[0 : self.H : step, 0 : self.W : step]
        y, x = y.flatten(), x.flatten()
        colors = img[y, x][:, [2, 1, 0]]
        dx, dy = x - self.cx, y - self.cy
        r_pixel = np.sqrt(dx**2 + dy**2)
        mask = r_pixel <= (self.R * 0.995)
        valid_r, valid_dx, valid_dy = r_pixel[mask], dx[mask], dy[mask]
        phi = np.arctan2(valid_dy, valid_dx)
        theta = valid_r / self.f
        vx, vy, vz = (
            np.sin(theta) * np.cos(phi),
            np.sin(theta) * np.sin(phi),
            np.cos(theta),
        )
        return np.column_stack((vx, vy, vz)).astype(np.float32), colors[mask]
