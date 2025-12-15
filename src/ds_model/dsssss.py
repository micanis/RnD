import cv2
import numpy as np
import struct
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional

# ==========================================
# 1. Domain Primitive & Utilities
# ==========================================

@dataclass
class Point3D:
    x: float
    y: float
    z: float

    def to_array(self) -> np.ndarray:
        return np.array([self.x, self.y, self.z], dtype=np.float32)

class VectorMath:
    """ベクトル演算の共通ロジック"""
    @staticmethod
    def normalize(v: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(v)
        if norm < 1e-10: return v
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
        """
        ロドリゲスの回転公式を用いて、ベクトルvをaxis周りに回転させる
        (vとaxisは直交している前提の簡易版: n' = n cosθ + (e x n) sinθ)
        """
        rad = np.radians(angle_deg)
        cos_theta = np.cos(rad)
        sin_theta = np.sin(rad)
        
        # axis(肘)とv(法線)は直交しているため、第3項(k(k・v)(1-cos))は0になる
        cross_term = np.cross(axis, v)
        
        return v * cos_theta + cross_term * sin_theta

# ==========================================
# 2. Interfaces (Abstractions)
# ==========================================

class ICameraModel(ABC):
    @abstractmethod
    def pixel_to_vector(self, u: float, v: float) -> np.ndarray:
        pass

    @abstractmethod
    def generate_background_cloud(self, img: np.ndarray, step: int) -> Tuple[np.ndarray, np.ndarray]:
        pass

class IPointCloudWriter(ABC):
    @abstractmethod
    def write(self, filepath: str, points: np.ndarray, colors: np.ndarray):
        pass

# ==========================================
# 3. Implementations (Details)
# ==========================================

class FisheyeCamera(ICameraModel):
    def __init__(self, width: int, height: int, fov_deg: float = 180.0):
        self.W = width
        self.H = height
        self.cx = width / 2.0
        self.cy = height / 2.0
        self.R = min(width, height) / 2.0
        fov_rad = np.radians(fov_deg)
        self.f = self.R / (fov_rad / 2.0)

    def pixel_to_vector(self, u: float, v: float) -> np.ndarray:
        dx = u - self.cx
        dy = v - self.cy
        r = np.sqrt(dx**2 + dy**2)
        phi = np.arctan2(dy, dx)
        theta = r / self.f
        
        vx = np.sin(theta) * np.cos(phi)
        vy = np.sin(theta) * np.sin(phi)
        vz = np.cos(theta)
        return np.array([vx, vy, vz], dtype=np.float32)

    def generate_background_cloud(self, img: np.ndarray, step: int = 4) -> Tuple[np.ndarray, np.ndarray]:
        y, x = np.mgrid[0:self.H:step, 0:self.W:step]
        y = y.flatten()
        x = x.flatten()
        colors = img[y, x][:, [2, 1, 0]] 

        dx = x - self.cx
        dy = y - self.cy
        r_pixel = np.sqrt(dx**2 + dy**2)
        mask = r_pixel <= (self.R * 0.995)
        
        valid_r = r_pixel[mask]
        valid_dx = dx[mask]
        valid_dy = dy[mask]
        valid_colors = colors[mask]
        
        phi = np.arctan2(valid_dy, valid_dx)
        theta = valid_r / self.f
        
        vx = np.sin(theta) * np.cos(phi)
        vy = np.sin(theta) * np.sin(phi)
        vz = np.cos(theta)
        
        points = np.column_stack((vx, vy, vz)).astype(np.float32)
        return points, valid_colors

class BinaryPLYWriter(IPointCloudWriter):
    def write(self, filepath: str, points: np.ndarray, colors: np.ndarray):
        points = points.astype(np.float32)
        colors = colors.astype(np.uint8)
        n_points = len(points)
        
        header = f"""ply
format binary_little_endian 1.0
element vertex {n_points}
property float x
property float y
property float z
property uchar red
property uchar green
property uchar blue
end_header
"""
        with open(filepath, 'wb') as f:
            f.write(header.encode('ascii'))
            vertex_dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'), 
                            ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
            vertices = np.empty(n_points, dtype=vertex_dtype)
            vertices['x'] = points[:, 0]
            vertices['y'] = points[:, 1]
            vertices['z'] = points[:, 2]
            vertices['red'] = colors[:, 0]
            vertices['green'] = colors[:, 1]
            vertices['blue'] = colors[:, 2]
            f.write(vertices.tobytes())

class GeometryBuilder:
    @staticmethod
    def create_block_marker(camera: ICameraModel, u: float, v: float, color: List[int], 
                            pixel_step: int = 3, grid_size: int = 1, radius_offset: float = 1.02) -> Tuple[List, List]:
        points = []
        colors = []
        range_val = range(-grid_size, grid_size + 1)
        for dy in range_val:
            for dx in range_val:
                u_new = u + dx * pixel_step
                v_new = v + dy * pixel_step
                vec = camera.pixel_to_vector(u_new, v_new)
                vec = VectorMath.normalize(vec) * radius_offset
                points.append(vec)
                colors.append(color)
        return points, colors

    @staticmethod
    def create_great_circle(normal: np.ndarray, color: List[int], num_points: int = 1000, radius_offset: float = 1.01) -> Tuple[List, List]:
        if abs(normal[2]) < 0.9:
            u_axis = VectorMath.normalize(np.cross(normal, np.array([0, 0, 1])))
        else:
            u_axis = VectorMath.normalize(np.cross(normal, np.array([1, 0, 0])))
        v_axis = VectorMath.normalize(np.cross(normal, u_axis))
        
        points = []
        cols = []
        for theta in np.linspace(0, 2 * np.pi, num_points):
            p = u_axis * np.cos(theta) + v_axis * np.sin(theta)
            points.append(p * radius_offset)
            cols.append(color)
        return points, cols

    @staticmethod
    def create_line(vector: np.ndarray, color: List[int], length: float = 1.2, num_points: int = 100) -> Tuple[List, List]:
        points = []
        cols = []
        for r in np.linspace(0, length, num_points):
            points.append(vector * r)
            cols.append(color)
        return points, cols

# ==========================================
# 4. Use Case / Application Logic
# ==========================================

class PointingVisualizer:
    
    def __init__(self, camera: ICameraModel, writer: IPointCloudWriter, output_dir: str):
        self.camera = camera
        self.writer = writer
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def process_background(self, img: np.ndarray, filename: str = "0_background_hemisphere.ply"):
        print("Generating Background...")
        pts, cols = self.camera.generate_background_cloud(img, step=4)
        self._save(filename, pts, cols)

    def process_scene(self, elbow_uv: Tuple[float, float], wrist_uv: Tuple[float, float], targets_uv: Dict[str, Tuple[float, float]]):
        print("Processing Scene Geometry...")
        
        v_elbow = VectorMath.normalize(self.camera.pixel_to_vector(*elbow_uv))
        v_wrist = VectorMath.normalize(self.camera.pixel_to_vector(*wrist_uv))
        
        # 法線計算 (メインの指差し平面)
        normal = VectorMath.cross_product_normalized(v_elbow, v_wrist)
        
        self._generate_arm_points(elbow_uv, wrist_uv)
        
        # メインの大円 (緑)
        self._generate_great_circle(normal, "2_great_circle_center.ply", [0, 255, 0])
        
        # 左右15度の境界線 (黄色) [New!]
        self._generate_boundary_lines(normal, v_elbow)

        self._generate_normal_vector(normal)
        self._generate_targets(normal, v_elbow, v_wrist, targets_uv)
        
    def _generate_arm_points(self, elbow_uv, wrist_uv):
        pts_list = []
        cols_list = []
        p, c = GeometryBuilder.create_block_marker(self.camera, *elbow_uv, [0, 0, 255])
        pts_list.extend(p); cols_list.extend(c)
        p, c = GeometryBuilder.create_block_marker(self.camera, *wrist_uv, [0, 255, 255])
        pts_list.extend(p); cols_list.extend(c)
        self._save("1_arm_points.ply", pts_list, cols_list)

    def _generate_great_circle(self, normal, filename, color):
        pts, cols = GeometryBuilder.create_great_circle(normal, color)
        self._save(filename, pts, cols)
    
    def _generate_boundary_lines(self, normal: np.ndarray, axis: np.ndarray):
        """法線を軸周りに±15度回転させて境界線を描画する"""
        # +15度回転した法線 -> 境界線1
        normal_plus = VectorMath.rotate_around_axis(normal, axis, 15.0)
        pts_p, cols_p = GeometryBuilder.create_great_circle(normal_plus, [255, 255, 0]) # 黄色
        
        # -15度回転した法線 -> 境界線2
        normal_minus = VectorMath.rotate_around_axis(normal, axis, -15.0)
        pts_m, cols_m = GeometryBuilder.create_great_circle(normal_minus, [255, 255, 0]) # 黄色
        
        # まとめて保存
        self._save("2_great_circle_bounds.ply", pts_p + pts_m, cols_p + cols_m)

    def _generate_normal_vector(self, normal):
        pts, cols = GeometryBuilder.create_line(normal, [255, 255, 255])
        self._save("3_normal_vector.ply", pts, cols)

    def _generate_targets(self, normal, v_elbow, v_wrist, targets_uv):
        pts_list = []
        cols_list = []
        
        for name, uv in targets_uv.items():
            v_tgt = VectorMath.normalize(self.camera.pixel_to_vector(*uv))
            
            # 判定ロジック
            angle_diff = VectorMath.angle_between(normal, v_tgt)
            deviation = abs(90.0 - angle_diff)
            
            score_fwd = np.dot(v_tgt, v_wrist) - np.dot(v_tgt, v_elbow)
            
            # 許容範囲を15度に拡大
            is_hit = (deviation < 15.0) and (score_fwd > 0)
            
            print(f"Target {name}: Dev={deviation:.1f}deg -> {'HIT' if is_hit else 'MISS'}")
            
            color = [255, 255, 0] if is_hit else [255, 0, 0]
            p, c = GeometryBuilder.create_block_marker(self.camera, *uv, color)
            pts_list.extend(p); cols_list.extend(c)
            
        self._save("4_targets.ply", pts_list, cols_list)

    def _save(self, filename: str, points: list | np.ndarray, colors: list | np.ndarray):
        path = os.path.join(self.output_dir, filename)
        if isinstance(points, list): points = np.array(points)
        if isinstance(colors, list): colors = np.array(colors)
        self.writer.write(path, points, colors)
        print(f"Saved: {path}")

# ==========================================
# 5. Client Code
# ==========================================

if __name__ == "__main__":
    INPUT_IMAGE = "test_img.jpg"
    OUTPUT_DIR = "solid_vis_output"
    
    if not os.path.exists(INPUT_IMAGE):
        print(f"Error: {INPUT_IMAGE} not found.")
        dummy_img = np.zeros((1000, 1000, 3), dtype=np.uint8)
        cv2.circle(dummy_img, (500, 500), 490, (100, 100, 100), -1)
        img = dummy_img
        H, W = 1000, 1000
    else:
        img = cv2.imread(INPUT_IMAGE)
        H, W = img.shape[:2]

    camera_model = FisheyeCamera(width=W, height=H, fov_deg=180)
    ply_writer = BinaryPLYWriter()
    
    app = PointingVisualizer(camera_model, ply_writer, OUTPUT_DIR)
    app.process_background(img)
    
    # 座標データ入力
    elbow_pos = (274.8125, 749.26953125)
    wrist_pos = (269.828125, 793.94921875)
    targets = {
        "Light": (947, 1673),
        "AC": (344, 402),
        "TV": (663, 293)
    }
    
    app.process_scene(elbow_pos, wrist_pos, targets)
    print("\nProcessing Complete.")