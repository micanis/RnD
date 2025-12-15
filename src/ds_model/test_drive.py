import numpy as np
import os

# --- 1. 座標変換ロジック (Fisheye -> Unit Sphere) ---
def fisheye_to_sphere(u, v, img_width, img_height, fov_deg=180):
    cx = img_width / 2.0
    cy = img_height / 2.0
    R = min(img_width, img_height) / 2.0
    
    # 2D平面でのズレ
    dx = u - cx
    dy = v - cy
    r_pixel = np.sqrt(dx**2 + dy**2)
    phi = np.arctan2(dy, dx)
    
    # 入射角 theta (等距離射影 r = f * theta)
    fov_rad = np.radians(fov_deg)
    f = R / (fov_rad / 2.0)
    theta = r_pixel / f
    
    # 3Dベクトル化 (Z-forward)
    vec_x = np.sin(theta) * np.cos(phi)
    vec_y = np.sin(theta) * np.sin(phi)
    vec_z = np.cos(theta)
    
    return np.array([vec_x, vec_y, vec_z])

def normalize(v):
    norm = np.linalg.norm(v)
    if norm < 1e-10: return v
    return v / norm

# --- 2. メイン処理: 判定と可視化ファイルの生成 ---
def process_pointing_visualization(elbow_uv, wrist_uv, targets_uv, image_size, output_dir="vis_result"):
    """
    Args:
        elbow_uv (tuple): 肘の座標 (u, v)
        wrist_uv (tuple): 手首の座標 (u, v)
        targets_uv (dict): 目標点の辞書 {'name': (u, v), ...}
        image_size (tuple): (width, height)
    """
    os.makedirs(output_dir, exist_ok=True)
    W, H = image_size

    print(f"--- Processing Image size: {W}x{H} ---")

    # 1. ベクトル変換
    v_elbow = normalize(fisheye_to_sphere(*elbow_uv, W, H))
    v_wrist = normalize(fisheye_to_sphere(*wrist_uv, W, H))
    v_origin = np.array([0.0, 0.0, 0.0])

    print(f"Elbow Vec: {v_elbow}")
    print(f"Wrist Vec: {v_wrist}")

    # 2. 平面の法線ベクトル算出 (Cross Product)
    # これが「指差し平面」の向き
    normal = normalize(np.cross(v_elbow, v_wrist))

    # 3. ターゲットの判定と点群作成
    target_points = []
    target_colors = []

    print("\n--- Target Evaluation ---")
    for name, uv in targets_uv.items():
        v_target = normalize(fisheye_to_sphere(*uv, W, H))
        
        # 判定: 法線とターゲットの内積をとる
        # 0 (直角) に近いほど、平面上にある
        dot_val = np.clip(np.dot(normal, v_target), -1.0, 1.0)
        angle_diff = np.degrees(np.arccos(dot_val))
        deviation = abs(90.0 - angle_diff) # 90度からのズレ
        
        # 前方判定 (手首側にあるか、肘の背中側か)
        # 手首との距離(内積)の方が、肘との距離より近ければ前方
        score_fwd = np.dot(v_target, v_wrist) - np.dot(v_target, v_elbow)
        is_forward = score_fwd > 0

        # 色分け判定
        # ズレが5度以内 かつ 前方なら「HIT (黄色)」
        # それ以外は「MISS (赤)」
        is_hit = (deviation < 5.0) and is_forward
        
        print(f"Target '{name}': Dev={deviation:.2f} deg, Fwd={is_forward} -> {'HIT!' if is_hit else 'MISS'}")

        target_points.append(v_target)
        if is_hit:
            target_colors.append([255, 255, 0]) # Yellow
        else:
            target_colors.append([255, 0, 0])   # Red

    # --- 4. 可視化用データの生成 (前回と同じロジック) ---
    
    # (A) 腕 (白->青->水色)
    pts_arm = np.vstack([v_origin, v_elbow, v_wrist])
    cols_arm = np.array([[255, 255, 255], [0, 0, 255], [0, 255, 255]])

    # (B) 平面/大円 (Green Zone)
    if abs(normal[2]) < 0.9:
        u_axis = normalize(np.cross(normal, np.array([0,0,1])))
    else:
        u_axis = normalize(np.cross(normal, np.array([1,0,0])))
    v_axis = normalize(np.cross(normal, u_axis))
    
    plane_pts = []
    plane_cols = []
    for theta in np.linspace(0, 2*np.pi, 720): # 細かく
        p_circle = u_axis * np.cos(theta) + v_axis * np.sin(theta)
        # 半径方向に線を引く感じにする
        for r in np.linspace(0.1, 1.0, 5):
            plane_pts.append(p_circle * r)
            plane_cols.append([0, 255, 0]) # Green

    # (C) 法線 (White Line)
    normal_pts = [normal * r for r in np.linspace(0, 1.2, 50)]
    normal_cols = [[255, 255, 255]] * len(normal_pts)

    # --- 5. 保存 ---
    save_ply(os.path.join(output_dir, "1_arm.ply"), pts_arm, cols_arm)
    save_ply(os.path.join(output_dir, "2_plane.ply"), np.array(plane_pts), np.array(plane_cols))
    save_ply(os.path.join(output_dir, "3_normal.ply"), np.array(normal_pts), np.array(normal_cols))
    save_ply(os.path.join(output_dir, "4_targets.ply"), np.array(target_points), np.array(target_colors))
    
    print(f"\nVisualization saved to '{output_dir}/'")

def save_ply(filename, points, colors):
    header = f"""ply
format ascii 1.0
element vertex {len(points)}
property float x
property float y
property float z
property uchar red
property uchar green
property uchar blue
end_header
"""
    with open(filename, 'w') as f:
        f.write(header)
        for p, c in zip(points, colors):
            f.write(f"{p[0]:.4f} {p[1]:.4f} {p[2]:.4f} {int(c[0])} {int(c[1])} {int(c[2])}\n")

# ==========================================
# ここにあなたの実際の座標を入力してください
# ==========================================
if __name__ == "__main__":
    # 画像サイズ (ThetaVならリサイズ後などを想定)
    W, H = 1500, 1500 
    
    # 検出された座標 (u, v)
    # ここでは仮の値を入れますが、あなたのデータを代入してください
    
    # 例: 画面右下あたりで、中心方向へ腕を伸ばしている
    pt_elbow = (755, 1228) 
    pt_wrist = (535, 1085)
    
    # 目標点リスト
    targets = {
        "Target_A": (344, 402), # 延長線上にあるはず (HIT期待)
        "Target_B": (974, 1673), # 全然違う方向 (MISS期待)
        "Target_C": (663, 293)  # 微妙に近い (HIT/MISS境界)
    }

    # 実行
    process_pointing_visualization(pt_elbow, pt_wrist, targets, (W, H))