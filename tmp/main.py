import cv2
import numpy as np
import zarr

from detector import detect_person
from distortion_correction import extract_view, get_theta_z1_params


def pixel_to_pan_tilt(cx, cy, img_width, img_height, fov_deg=183.0):
    """
    画像上のピクセル座標からpan/tilt角度を計算する（等距離射影モデル）
    """
    K, _ = get_theta_z1_params(img_width, img_height, fov_deg)
    f = K[0, 0]
    img_cx = K[0, 2]
    img_cy = K[1, 2]

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

    return pan, tilt


def main():
    store = zarr.ZipStore("data/processed/image/dual_fisheye/test2.zarr.zip", mode="r")
    root = zarr.group(store=store)

    img = root["right"][0]
    img = np.asarray(img)

    bbox = detect_person(img)
    if bbox is None:
        print("人物が検出されませんでした")
        store.close()
        return

    x1, y1, x2, y2 = bbox
    person_cx = (x1 + x2) / 2
    person_cy = (y1 + y2) / 2

    h, w = img.shape[:2]
    pan, tilt = pixel_to_pan_tilt(person_cx, person_cy, w, h)
    print(
        f"検出位置: ({person_cx:.0f}, {person_cy:.0f}) -> pan={pan:.1f}°, tilt={tilt:.1f}°"
    )

    if img.shape[2] == 3:
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    else:
        img_bgr = img

    result = extract_view(img_bgr, pan=pan, tilt=tilt)

    cv2.imwrite("tmp/result.jpg", result)
    print("保存しました: tmp/result.jpg")

    store.close()


if __name__ == "__main__":
    main()
