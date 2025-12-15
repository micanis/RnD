import cv2
import numpy as np
import struct
import os

def generate_high_quality_dual_spheres(img_path, xi, output_dir="output_hq_spheres"):
    """
    単眼魚眼画像のフル解像度データから、高速に「外側の球(q)」と「内側の球(s)」の
    バイナリPLYファイルを生成する。
    """
    # 出力ディレクトリ作成
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}")

    # 1. 画像読み込み (フル解像度)
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"Image not found: {img_path}")
    
    h, w = img.shape[:2]
    print(f"Input Resolution: {w}x{h}")

    # 2. レンズパラメータ設定 (単眼中心)
    cx, cy = w / 2.0, h / 2.0
    R = min(w, h) / 2.0 
    fov_rad = np.pi # 180度

    # 3. メッシュグリッドとマスク作成 (ベクトル化準備)
    # メモリ効率のため、ここで有効な画素だけを抽出するマスクを作ります
    x = np.arange(w, dtype=np.float32)
    y = np.arange(h, dtype=np.float32)
    xv, yv = np.meshgrid(x, y)

    dx = xv - cx
    dy = yv - cy
    r_pixel = np.sqrt(dx**2 + dy**2)
    
    # 円の外側をカット (境界ノイズ除去のため少しだけ内側で切る)
    mask = r_pixel <= (R * 0.995)
    
    # データを1次元化して抽出 (数百万点レベルの高速化の鍵)
    valid_r = r_pixel[mask]
    valid_dx = dx[mask]
    valid_dy = dy[mask]
    
    # 色情報の抽出 (BGR -> RGB) と型変換
    colors = img[mask][:, [2, 1, 0]].astype(np.uint8)
    
    n_points = len(valid_r)
    print(f"Processing {n_points} points (Vectorized)...")

    # --- 共通計算: 角度 ---
    valid_phi = np.arctan2(valid_dy, valid_dx)
    
    # 等距離射影モデル: theta = r / f
    f = R / (fov_rad / 2.0)
    theta_incident = valid_r / f 
    
    sin_theta = np.sin(theta_incident)
    cos_theta = np.cos(theta_incident)

    # ==========================================================
    # Calculation 1: Outer Sphere (q) - 元の光線方向
    # ==========================================================
    # Z軸を光軸(カメラ正面)とする座標系
    q_x = sin_theta * np.cos(valid_phi)
    q_y = sin_theta * np.sin(valid_phi)
    q_z = cos_theta

    # 結合して (N, 3) のfloat32配列にする
    points_outer = np.column_stack((q_x, q_y, q_z)).astype(np.float32)

    # ==========================================================
    # Calculation 2: Inner Sphere (s) - DSモデル(xi)適用後
    # ==========================================================
    # p = q + [0, 0, xi] (光軸Z方向にシフト)
    p_z = q_z + xi
    # p_x, p_y は q_x, q_y と同じ
    
    # 正規化: s = p / ||p||
    p_norm_sq = q_x**2 + q_y**2 + p_z**2
    p_norm = np.sqrt(p_norm_sq)
    # ゼロ除算回避
    p_norm[p_norm < 1e-8] = 1.0
    
    s_x = q_x / p_norm
    s_y = q_y / p_norm
    s_z = p_z / p_norm
    
    # 結合して (N, 3) のfloat32配列にする
    points_inner = np.column_stack((s_x, s_y, s_z)).astype(np.float32)

    # ==========================================================
    # 保存処理 (バイナリPLY)
    # ==========================================================
    outer_filename = os.path.join(output_dir, "outer_sphere.ply")
    inner_filename = os.path.join(output_dir, "inner_sphere.ply")

    print(f"Saving Outer Sphere (xi=0) to {outer_filename}...")
    save_ply_binary(outer_filename, points_outer, colors)
    
    print(f"Saving Inner Sphere (xi={xi}) to {inner_filename}...")
    save_ply_binary(inner_filename, points_inner, colors)
    
    print("Done!")

def save_ply_binary(filename, points, colors):
    """
    高速・軽量なバイナリ形式でPLYを保存する関数
    """
    n_points = len(points)
    
    # PLYヘッダー
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
    
    with open(filename, 'wb') as f:
        f.write(header.encode('ascii'))
        
        # 構造化配列を使って頂点と色をインターリーブ結合
        vertex_dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'), 
                        ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
        
        vertices = np.empty(n_points, dtype=vertex_dtype)
        vertices['x'] = points[:, 0]
        vertices['y'] = points[:, 1]
        vertices['z'] = points[:, 2]
        vertices['red'] = colors[:, 0]
        vertices['green'] = colors[:, 1]
        vertices['blue'] = colors[:, 2]
        
        # バイナリ一括書き込み
        f.write(vertices.tobytes())

# --- 実行設定 ---
# 180度分の正方形に近い魚眼画像を用意してください
input_img = "test_img.jpg" 
xi_param = 0.5 # DSモデルのパラメータ

# 実行
try:
    # output_hq_spheres というフォルダに2つのファイルが生成されます
    generate_high_quality_dual_spheres(input_img, xi_param)
except Exception as e:
    print(f"Error: {e}")