import cv2
import numpy as np
from pathlib import Path
import glob
from utils.paths import PATHS
from utils.sampling import sample_frames

# === 設定 ===
CHECKERBOARD = (4, 7)   # 内部コーナー数
SQUARE_SIZE = 34.0      # mm
IMAGE_DIR = Path(f"{PATHS.output}/from_video/calibration/left")

# === コーナー検出精度 ===
subpix_criteria = (
    cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
    30,
    0.1
)

# === キャリブレーション設定 ===
calibration_flags = (
    cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC +
    cv2.fisheye.CALIB_FIX_SKEW
)

# === チェスボードの3D点（単位：mm） ===
objp = np.zeros((1, CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
objp[0, :, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
objp *= SQUARE_SIZE

objpoints = []  # 3D点
imgpoints = []  # 2D点

# === 画像読み込み ===
images = sample_frames(IMAGE_DIR, 15)
print(f"🔍 {len(images)} 枚の画像を検出")

for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD,
                                             cv2.CALIB_CB_ADAPTIVE_THRESH +
                                             cv2.CALIB_CB_FAST_CHECK +
                                             cv2.CALIB_CB_NORMALIZE_IMAGE)

    if ret:
        # コーナーをサブピクセル精度に補正
        cv2.cornerSubPix(gray, corners, (3, 3), (-1, -1), subpix_criteria)
        imgpoints.append(corners)
        objpoints.append(objp)
        print(f"✅ 検出成功: {Path(fname).name}")
    else:
        print(f"⚠️ 失敗: {Path(fname).name}")

print(f"success corner: {len(objpoints)}")
if len(objpoints) < 3:
    raise RuntimeError("キャリブレーションに十分な画像がありません。")

# === キャリブレーション実行 ===
K = np.zeros((3, 3))
D = np.zeros((4, 1))
rvecs = []
tvecs = []

print("\n📷 カメラキャリブレーション中...")
rms, _, _, _, _ = cv2.fisheye.calibrate(
    objpoints,
    imgpoints,
    gray.shape[::-1],
    K,
    D,
    rvecs,
    tvecs,
    calibration_flags,
    (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-6)
)

print("\n=== 結果 ===")
print(f"RMS誤差: {rms:.4f}")
print("カメラ行列 K:")
print(K)
print("\n歪み係数 D:")
print(D.ravel())

# === 結果を保存 ===
output_file = IMAGE_DIR.parent / "calibration_result.npz"
np.savez(output_file, K=K, D=D, rms=rms)
print(f"\n💾 保存しました → {output_file}")
