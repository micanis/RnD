from dataclasses import dataclass

import cv2
import numpy as np
from scipy.spatial.transform import Rotation


@dataclass
class CameraParams:
    """カメラ内部パラメータ"""

    K: np.ndarray  # 3x3 カメラ行列
    D: np.ndarray  # 歪み係数


class FisheyeCorrector:
    """魚眼レンズの歪み補正・仮想PTZ"""

    def __init__(self, width: int, height: int, fov_deg: float = 183.0):
        """
        Parameters
        ----------
        width : int
            入力画像の幅
        height : int
            入力画像の高さ
        fov_deg : float
            レンズの視野角（度）。THETA Z1は約183度
        """
        self.width = width
        self.height = height
        self.fov_deg = fov_deg
        self.params = self._calc_params()

    def _calc_params(self) -> CameraParams:
        """カメラパラメータを計算"""
        cx = self.width / 2.0
        cy = self.height / 2.0

        radius_px = min(self.width, self.height) / 2.0
        theta_max_rad = np.radians(self.fov_deg / 2.0)

        f = radius_px / theta_max_rad

        K = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]], dtype=np.float32)
        D = np.zeros((4, 1), dtype=np.float32)

        return CameraParams(K=K, D=D)

    def extract_view(
        self,
        img: np.ndarray,
        pan: float,
        tilt: float,
        out_size: tuple[int, int] = (640, 480),
        fov: float = 90.0,
    ) -> np.ndarray:
        """
        指定方向の透視投影画像を生成（仮想PTZ）

        Parameters
        ----------
        img : np.ndarray
            入力魚眼画像
        pan : float
            水平角（度）。左がマイナス、右がプラス
        tilt : float
            垂直角（度）。下がマイナス、上がプラス
        out_size : tuple[int, int]
            出力画像サイズ (width, height)
        fov : float
            切り出す視野角（度）。小さいほどズームイン

        Returns
        -------
        np.ndarray
            透視投影画像
        """
        R = (
            Rotation.from_euler("yx", [pan, tilt], degrees=True)
            .as_matrix()
            .astype(np.float32)
        )

        f_out = (out_size[0] / 2) / np.tan(np.radians(fov / 2))
        K_new = np.array(
            [[f_out, 0, out_size[0] / 2], [0, f_out, out_size[1] / 2], [0, 0, 1]],
            dtype=np.float32,
        )

        map1, map2 = cv2.fisheye.initUndistortRectifyMap(
            self.params.K, self.params.D, R, K_new, out_size, cv2.CV_16SC2
        )

        return cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR)

    def pixel_to_angle(self, cx: float, cy: float) -> tuple[float, float]:
        """
        ピクセル座標をpan/tilt角度に変換（等距離射影モデル）

        Parameters
        ----------
        cx : float
            人物中心のx座標（ピクセル）
        cy : float
            人物中心のy座標（ピクセル）

        Returns
        -------
        tuple[float, float]
            (pan, tilt) 度単位
        """
        f = self.params.K[0, 0]
        img_cx = self.params.K[0, 2]
        img_cy = self.params.K[1, 2]

        dx = cx - img_cx
        dy = cy - img_cy

        r = np.sqrt(dx**2 + dy**2)
        theta = r / f

        if r > 0:
            pan = np.degrees(-theta * dx / r)
            tilt = np.degrees(theta * dy / r)
        else:
            pan = 0.0
            tilt = 0.0

        return float(pan), float(tilt)

    def extract_view_from_bbox(
        self,
        img: np.ndarray,
        bbox: np.ndarray,
        out_size: tuple[int, int] = (640, 480),
        fov: float = 90.0,
    ) -> np.ndarray:
        """
        bboxから透視投影画像を生成（仮想PTZ）

        bboxの中心座標をpan/tiltに変換し、extract_viewを呼び出す。

        Parameters
        ----------
        img : np.ndarray
            入力魚眼画像
        bbox : np.ndarray
            バウンディングボックス [x1, y1, x2, y2]
        out_size : tuple[int, int]
            出力画像サイズ (width, height)
        fov : float
            切り出す視野角（度）。小さいほどズームイン

        Returns
        -------
        np.ndarray
            透視投影画像
        """
        cx = (bbox[0] + bbox[2]) / 2.0
        cy = (bbox[1] + bbox[3]) / 2.0
        pan, tilt = self.pixel_to_angle(cx, cy)
        return self.extract_view(img, pan, tilt, out_size, fov)
