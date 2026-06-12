## 目的

魚眼画像で人物を安定して検出できるモデルを比較し、実環境で利用する候補モデルを選定する。

## 評価データ

- 評価用データセット
- 実環境で撮影した魚眼画像

## 評価指標

1. 検出精度 - 検出対象の人物を正しく検出できているか
2. 領域一致度 - Ground Truth と検出領域がどの程度重なっているか
3. 推論速度 - 実環境で必要な処理速度を満たせるか
4. モデルサイズ - 配置先の計算資源に収まるか

## 比較対象モデル

- YOLOv8 以降 - 透視投影画像を主な対象とする汎用物体検出モデル
- RAPID - 魚眼画像に特化した人物検出モデル

## 実験条件

- CEPDOF: `data/raw/CEPDOF/Lunch1` の 0-based 500 フレーム目
- 実環境: `data/processed/image/dual_fisheye/test2.zarr.zip` の `right[0]`
- YOLO: `models/yolov8m.pt`
- RAPiD: `workspace/src/RAPiD/weights/pL1_MWHB1024_Mar11_4000.ckpt`
- RAPiD bbox 形式: `[center_x, center_y, width, height, angle, confidence]`

## 実験結果

| モデル | 画像 | 検出数 | 推論時間 | モデルサイズ | 出力 |
| --- | --- | ---: | ---: | ---: | --- |
| YOLOv8m | `cepdof_lunch1_0500.jpg` | 3 | 216.0 ms | 49.7 MB | `outputs/yolo/cepdof_lunch1_0500_yolo.jpg` |
| YOLOv8m | `real_test2_000.jpg` | 4 | 143.3 ms | 49.7 MB | `outputs/yolo/real_test2_000_yolo.jpg` |
| RAPiD | `cepdof_lunch1_0500.jpg` | 2 | 229.6 ms | 235.0 MB | `outputs/rapid/cepdof_lunch1_0500_rapid.jpg` |
| RAPiD | `real_test2_000.jpg` | 2 | 46.3 ms | 235.0 MB | `outputs/rapid/real_test2_000_rapid.jpg` |

| モデル | 平均検出数 | 平均推論時間 | モデルサイズ |
| --- | ---: | ---: | ---: |
| YOLOv8m | 3.5 | 179.6 ms | 49.7 MB |
| RAPiD | 2.0 | 138.0 ms | 235.0 MB |

## 評価

### 検出精度

今回の2枚の評価では Ground Truth bbox を用意していないため、厳密な Precision / Recall / IoU は算出していない。検出数と confidence、および描画結果の目視確認をもとに比較する。

- YOLOv8m は CEPDOF で 3 件、実環境で 4 件を検出した。
- RAPiD は CEPDOF と実環境の両方で 2 件を検出した。
- RAPiD の検出 confidence は高く、CEPDOF では `0.990`, `0.932`、実環境では `0.991`, `0.919` だった。
- YOLOv8m は低 confidence の検出も含めて候補を多く出す傾向があり、実環境では `0.294` の検出も含まれていた。

### 領域一致度

- YOLOv8m は axis-aligned bbox を出力するため、魚眼画像で斜め方向に写る人物に対して bbox が広がりやすい。
- RAPiD は rotated bbox を出力するため、魚眼画像上の人物姿勢に合わせた領域表現ができる。
- Ground Truth が無いため IoU による定量評価は未実施だが、魚眼画像向けの領域表現としては RAPiD の方が適している。

### 推論速度

- 平均推論時間は YOLOv8m が `179.6 ms`、RAPiD が `138.0 ms` だった。
- 実環境画像では RAPiD が `46.3 ms` と速く、YOLOv8m の `143.3 ms` より大きく短い。
- CEPDOF 画像では RAPiD が `229.6 ms`、YOLOv8m が `216.0 ms` で、ほぼ同程度だった。

### モデルサイズ

- YOLOv8m は `49.7 MB`。
- RAPiD は `235.0 MB`。
- 配置先のストレージやメモリ制約を重視する場合は YOLOv8m の方が扱いやすい。

## 結論

実環境で魚眼画像の人物領域を安定して扱う候補としては RAPiD を優先する。理由は、魚眼画像に合わせた rotated bbox を出力でき、今回の実環境画像では YOLOv8m より高速だったためである。

一方で、RAPiD はモデルサイズが大きく、検出数も YOLOv8m より少なかった。人物の取りこぼしを避けたい用途や、軽量な配置を優先する用途では YOLOv8m も候補として残す。最終判断には Ground Truth bbox を用意し、IoU、Precision、Recall を追加で評価する必要がある。
