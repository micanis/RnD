import numpy as np
from typing import List, Tuple

from .math_utils import VectorMath


class GeometryBuilder:
    @staticmethod
    def create_block_marker(camera, u: float, v: float, color: List[int], pixel_step: int = 3, grid_size: int = 1, radius_offset: float = 1.02) -> Tuple[List, List]:
        points, colors = [], []
        for dy in range(-grid_size, grid_size + 1):
            for dx in range(-grid_size, grid_size + 1):
                vec = camera.pixel_to_vector(u + dx * pixel_step, v + dy * pixel_step)
                points.append(VectorMath.normalize(vec) * radius_offset)
                colors.append(color)
        return points, colors

    @staticmethod
    def create_forward_arc(start_vec: np.ndarray, normal: np.ndarray, color: List[int], radius_offset: float = 1.01, step_deg: float = 0.5) -> Tuple[List, List]:
        points, cols = [], []
        tangent = VectorMath.normalize(np.cross(normal, start_vec))
        for deg in np.arange(0, 180, step_deg):
            rad = np.radians(deg)
            p = start_vec * np.cos(rad) + tangent * np.sin(rad)
            if p[2] < -0.05:
                break
            points.append(p * radius_offset)
            cols.append(color)
        return points, cols

    @staticmethod
    def create_line(vector: np.ndarray, color: List[int], length: float = 1.2, num_points: int = 100) -> Tuple[List, List]:
        points, cols = [], []
        for r in np.linspace(0, length, num_points):
            points.append(vector * r)
            cols.append(color)
        return points, cols
