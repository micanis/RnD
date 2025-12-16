import os
import numpy as np
import cv2
from typing import Dict, Tuple, List

from .math_utils import VectorMath
from .geometry_builder import GeometryBuilder
from .ply_io import load_ply_binary_simple


class PointingVisualizer:
    def __init__(self, camera, writer, output_dir: str):
        self.camera = camera
        self.writer = writer
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.original_image = None
        self.best_match_info = None

    def process_background(self, img: np.ndarray, filename: str = "0_background_hemisphere.ply"):
        self.original_image = img
        pts, cols = self.camera.generate_background_cloud(img, step=4)
        self._save(filename, pts, cols)

    def process_scene(self, elbow_uv: Tuple[float, float], wrist_uv: Tuple[float, float], targets_uv: Dict[str, Tuple[float, float]]):
        v_elbow = VectorMath.normalize(self.camera.pixel_to_vector(*elbow_uv))
        v_wrist = VectorMath.normalize(self.camera.pixel_to_vector(*wrist_uv))
        normal = VectorMath.cross_product_normalized(v_elbow, v_wrist)

        self._generate_arm_points(elbow_uv, wrist_uv)
        self.best_match_info = self._evaluate_all_targets_signed(normal, v_elbow, v_wrist, targets_uv)
        self._generate_forward_arc(v_elbow, normal, "2_great_circle_center.ply", [0, 255, 0])

        if self.best_match_info:
            signed_dev = self.best_match_info["signed_deviation"]
            self._generate_single_boundary_arc(v_elbow, normal, signed_dev)

        self._generate_normal_vector(normal)
        self._generate_target_points(self.best_match_info, targets_uv)

    def render_final_image(self, output_filename: str = "result.jpg"):
        W, H = self.camera.resolution
        canvas = self.original_image.copy() if self.original_image is not None else np.zeros((H, W, 3), dtype=np.uint8)

        ply_files = sorted([f for f in os.listdir(self.output_dir) if f.endswith(".ply") and not f.startswith("0_")])
        for ply_file in ply_files:
            try:
                points, colors = load_ply_binary_simple(os.path.join(self.output_dir, ply_file))
            except Exception:
                continue
            for pt, col in zip(points, colors):
                u, v = self.camera.vector_to_pixel(pt)
                if 0 <= u < W and 0 <= v < H:
                    cv2.circle(canvas, (int(u), int(v)), 2, (int(col[2]), int(col[1]), int(col[0])), -1)

        if self.best_match_info:
            signed_dev = self.best_match_info["signed_deviation"]
            tgt_uv = self.best_match_info["uv"]
            text_str = f"{signed_dev:+.1f} deg"
            text_pos = (int(tgt_uv[0]) + 15, int(tgt_uv[1]) + 5)
            cv2.putText(canvas, text_str, text_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4)
            cv2.putText(canvas, text_str, text_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        cv2.imwrite(os.path.join(self.output_dir, output_filename), canvas)

    # internal helpers
    def _evaluate_all_targets_signed(self, normal, v_elbow, v_wrist, targets_uv):
        candidates = []
        for name, uv in targets_uv.items():
            v_tgt = VectorMath.normalize(self.camera.pixel_to_vector(*uv))
            angle_deg = np.degrees(np.arccos(np.clip(np.dot(normal, v_tgt), -1.0, 1.0)))
            signed_deviation = 90.0 - angle_deg
            score_fwd = np.dot(v_tgt, v_wrist) - np.dot(v_tgt, v_elbow)
            if score_fwd > 0:
                candidates.append(
                    {"name": name, "uv": uv, "signed_deviation": signed_deviation, "abs_deviation": abs(signed_deviation)}
                )
        return min(candidates, key=lambda x: x["abs_deviation"]) if candidates else None

    def _generate_target_points(self, best_match, targets_uv):
        pts_list, cols_list = [], []
        best_name = best_match["name"] if best_match else None
        for name, uv in targets_uv.items():
            is_best = name == best_name
            color = [255, 255, 0] if is_best else [255, 0, 0]
            p, c = GeometryBuilder.create_block_marker(self.camera, *uv, color)
            pts_list.extend(p)
            cols_list.extend(c)
        self._save("4_targets.ply", pts_list, cols_list)

    def _generate_single_boundary_arc(self, axis_vec, normal, angle_deg):
        rotated_normal = VectorMath.rotate_around_axis(normal, axis_vec, angle_deg)
        pts, cols = GeometryBuilder.create_forward_arc(axis_vec, rotated_normal, [255, 255, 0])
        self._save("2_great_circle_bounds.ply", pts, cols)

    def _generate_arm_points(self, elbow_uv, wrist_uv):
        pts_list, cols_list = [], []
        p, c = GeometryBuilder.create_block_marker(self.camera, *elbow_uv, [0, 0, 255])
        pts_list.extend(p)
        cols_list.extend(c)
        p, c = GeometryBuilder.create_block_marker(self.camera, *wrist_uv, [0, 255, 255])
        pts_list.extend(p)
        cols_list.extend(c)
        self._save("1_arm_points.ply", pts_list, cols_list)

    def _generate_forward_arc(self, start_vec, normal, filename, color):
        pts, cols = GeometryBuilder.create_forward_arc(start_vec, normal, color)
        self._save(filename, pts, cols)

    def _generate_normal_vector(self, normal):
        pts, cols = GeometryBuilder.create_line(normal, [255, 255, 255])
        self._save("3_normal_vector.ply", pts, cols)

    def _save(self, filename: str, points: List | np.ndarray, colors: List | np.ndarray):
        if isinstance(points, list):
            points = np.array(points)
        if isinstance(colors, list):
            colors = np.array(colors)
        self.writer.write(os.path.join(self.output_dir, filename), points, colors)
