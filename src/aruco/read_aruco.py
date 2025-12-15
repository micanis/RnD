import cv2
import glob
import os
import numpy as np

def main():
    # ---------------------------------------------------------
    # 設定：対象の画像ファイル名
    # ---------------------------------------------------------
    target_image_path = "test_img.jpg"  # ★ここを解析したい画像名に変えてください
    
    # ArUco検出器の準備 (4x4)
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    parameters = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(dictionary, parameters)

    # ---------------------------------------------------------
    # 手順1: 参照用画像から「IDと名前の対応表」を作る
    # ---------------------------------------------------------
    print("--- 参照マーカーの読み込み中 ---")
    
    # "marker_" で始まるjpgを探す
    base_dir = os.path.dirname(os.path.abspath(__file__))
    ref_images = glob.glob(os.path.join(base_dir, 'marker_*.jpg'))
    
    # { ID : "名前" } の辞書を作る
    registered_objects = {}

    for img_path in ref_images:
        # ファイル名から名前部分を抽出 (marker_00_TV.jpg -> TV)
        filename = os.path.basename(img_path)
        # "_"で分割して3つ目(インデックス2)を取り、".jpg"を削除
        # ファイル名が "marker_00_TV.jpg" 形式であることを想定
        try:
            obj_name = filename.split('_')[2].replace('.jpg', '')
        except IndexError:
            obj_name = filename # 形式が違う場合はファイル名そのまま

        # 画像を読み込んでIDを調べる
        ref_img = cv2.imread(img_path)
        if ref_img is None: continue
        
        gray_ref = cv2.cvtColor(ref_img, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = detector.detectMarkers(gray_ref)

        if ids is not None:
            marker_id = ids[0][0] # 最初の1個のIDを取得
            registered_objects[marker_id] = obj_name
            print(f"  登録完了: ID {marker_id} -> {obj_name}")
        else:
            print(f"  警告: {filename} からマーカーが見つかりませんでした")

    print(f"  -> 合計 {len(registered_objects)} 個のマーカー情報を登録しました。\n")

    # ---------------------------------------------------------
    # 手順2: ターゲット画像(1枚)から位置を特定する
    # ---------------------------------------------------------
    print(f"--- 画像解析: {target_image_path} ---")
    
    scene_img = cv2.imread(target_image_path)
    if scene_img is None:
        print("エラー: ターゲット画像が見つかりません。")
        return

    gray_scene = cv2.cvtColor(scene_img, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = detector.detectMarkers(gray_scene)

    if ids is not None:
        print(f"  検出されたマーカー総数: {len(ids)}\n")

        # 検出されたIDごとにループ
        for i, detected_id in enumerate(ids.flatten()):
            detected_id = int(detected_id)
            
            # 中心座標の計算
            pts = corners[i][0]
            cx = int(np.mean(pts[:, 0]))
            cy = int(np.mean(pts[:, 1]))

            # 登録リストにあるか確認
            if detected_id in registered_objects:
                obj_name = registered_objects[detected_id]
                print(f"  ★発見: [{obj_name}] (ID:{detected_id})")
                print(f"    座標: ({cx}, {cy})")
                
                # 画像に名前を描画
                label = f"{obj_name} ({cx},{cy})"
                cv2.putText(scene_img, label, (cx, cy - 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                # マーカーの枠を描画
                cv2.aruco.drawDetectedMarkers(scene_img, corners, ids)
            else:
                print(f"  未登録のマーカー (ID:{detected_id}) が見つかりました 座標:({cx}, {cy})")
    else:
        print("  マーカーは検出されませんでした。")

    # 結果表示
    cv2.imshow('Result', scene_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()