import cv2
import numpy as np
from scipy.spatial.transform import Rotation


def get_theta_z1_params(width, height, fov_deg=183.0):
    """
    THETA Z1のレンズ仕様からK, Dを推定する
    """
    cx = width / 2.0
    cy = height / 2.0

    # 対角線ではなく、円の半径（画像端）で計算
    # Z1の魚眼は画像の短辺（または正方形の一辺）に収まる設計
    radius_px = min(width, height) / 2.0
    theta_max_rad = np.radians(fov_deg / 2.0)

    # 焦点距離 f = r / theta
    f = radius_px / theta_max_rad

    K = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]], dtype=np.float32)

    # 歪み係数は0で近似（Z1のレンズは素直なのでこれでも十分綺麗です）
    D = np.zeros((4, 1), dtype=np.float32)

    return K, D


def extract_view(img, pan, tilt, out_size=(640, 480), fov_vptz=90):
    """
    指定方向にカメラを向けた透視投影画像を生成
    :param pan: 水平角 (度) - 左がマイナス、右がプラス
    :param tilt: 垂直角 (度) - 下がマイナス、上がプラス
    :param fov_vptz: 切り出す視野角（ズーム倍率に相当）
    """
    h, w = img.shape[:2]
    K, D = get_theta_z1_params(w, h)

    # 1. 回転行列の作成 (Pan, Tilt)
    # Z1のレンズ正面を基準とし、指定方向へ回転
    r = Rotation.from_euler("yx", [pan, tilt], degrees=True)
    R = r.as_matrix().astype(np.float32)

    # 2. 出力画像の内部パラメータ (仮想的なピンホールカメラ)
    f_out = (out_size[0] / 2) / np.tan(np.radians(fov_vptz / 2))
    K_new = np.array(
        [[f_out, 0, out_size[0] / 2], [0, f_out, out_size[1] / 2], [0, 0, 1]],
        dtype=np.float32,
    )

    # 3. マップ作成と適用
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(
        K, D, R, K_new, out_size, cv2.CV_16SC2
    )
    undistorted_img = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR)

    return undistorted_img


# --- メイン処理 ---
# THETA Z1のDual Fisheye画像を読み込み、左レンズ分だけ切り出したと想定
# raw_img = cv2.imread('theta_z1_dual_fisheye.jpg')
# left_lens = raw_img[:, :raw_img.shape[1]//2]

# 例えば、左斜め30度、上方向20度の位置にいる人物を正面として補正
# result = extract_view(left_lens, pan=-30, tilt=20)
# cv2.imshow("Virtual PTZ", result)
