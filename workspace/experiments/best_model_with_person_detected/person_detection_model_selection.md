## 目的

魚眼画像で人物を安定して検出できるモデルを比較し、実環境で利用する候補モデルを選定する。

## 評価データ

- 評価用データセット
- 実環境で撮影した魚眼画像

## 評価指標

1. 検出精度 - 検出対象の人物を正しく検出できているか
2. 推論速度 - 実環境で必要な処理速度を満たせるか
3. モデルサイズ - 配置先の計算資源に収まるか

## 比較対象モデル

- YOLOv8 以降 - 透視投影画像を主な対象とする汎用物体検出モデル
- RAPID - 魚眼画像に特化した人物検出モデル

## 実験条件

- CEPDOF: `data/raw/CEPDOF/Lunch1` の 0-based 500 フレーム目
- 実環境: `data/processed/image/dual_fisheye/test2.zarr.zip` の `right[0]`
- YOLO: `yolo26m.pt`, `yolo11m.pt`, `models/yolov8m.pt`
- Ultralytics: `8.4.64`
- RAPiD: `workspace/src/RAPiD/weights/pL1_MWHB1024_Mar11_4000.ckpt`
- RAPiD bbox 形式: `[center_x, center_y, width, height, angle, confidence]`
- 推論速度はモデルロード直後の初回実行を含めず、ウォームアップ後の複数回推論の平均値で比較する。
- デフォルト測定条件は `warmup_runs=2`, `measure_runs=5` とする。

## 実行方法

```bash
uv run yolo.py \
  --models yolo26m.pt yolo11m.pt models/yolov8m.pt \
  --warmup-runs 2 \
  --measure-runs 5

uv run rapid.py \
  --warmup-runs 2 \
  --measure-runs 5
```

## 実験結果

| モデル | 画像 | 検出数 | 推論時間 | モデルサイズ | 出力 |
| --- | --- | ---: | ---: | ---: | --- |
| YOLO26m | `cepdof_lunch1_0500.jpg` | 2 | 6.4 ms | 42.2 MB | `outputs/yolo/yolo26m/cepdof_lunch1_0500_yolo26m.jpg` |
| YOLO26m | `real_test2_000.jpg` | 2 | 6.3 ms | 42.2 MB | `outputs/yolo/yolo26m/real_test2_000_yolo26m.jpg` |
| YOLO11m | `cepdof_lunch1_0500.jpg` | 2 | 6.1 ms | 38.8 MB | `outputs/yolo/yolo11m/cepdof_lunch1_0500_yolo11m.jpg` |
| YOLO11m | `real_test2_000.jpg` | 2 | 6.2 ms | 38.8 MB | `outputs/yolo/yolo11m/real_test2_000_yolo11m.jpg` |
| YOLOv8m | `cepdof_lunch1_0500.jpg` | 3 | 5.2 ms | 49.7 MB | `outputs/yolo/yolov8m/cepdof_lunch1_0500_yolov8m.jpg` |
| YOLOv8m | `real_test2_000.jpg` | 4 | 5.2 ms | 49.7 MB | `outputs/yolo/yolov8m/real_test2_000_yolov8m.jpg` |
| RAPiD | `cepdof_lunch1_0500.jpg` | 2 | 49.0 ms | 235.0 MB | `outputs/rapid/cepdof_lunch1_0500_rapid.jpg` |
| RAPiD | `real_test2_000.jpg` | 2 | 45.4 ms | 235.0 MB | `outputs/rapid/real_test2_000_rapid.jpg` |

| モデル | 平均検出数 | 平均推論時間 | モデルサイズ |
| --- | ---: | ---: | ---: |
| YOLO26m | 2.0 | 6.4 ms | 42.2 MB |
| YOLO11m | 2.0 | 6.2 ms | 38.8 MB |
| YOLOv8m | 3.5 | 5.2 ms | 49.7 MB |
| RAPiD | 2.0 | 47.2 ms | 235.0 MB |

## 評価

### 検出精度

今回の2枚の評価では Ground Truth bbox を用意していないため、厳密な Precision / Recall / IoU は算出していない。検出数と confidence、および描画結果の目視確認をもとに比較する。

- YOLO26m は CEPDOF と実環境の両方で 2 件を検出した。
- YOLO11m は CEPDOF と実環境の両方で 2 件を検出した。
- YOLOv8m は CEPDOF で 3 件、実環境で 4 件を検出した。
- RAPiD は CEPDOF と実環境の両方で 2 件を検出した。
- RAPiD の検出 confidence は高く、CEPDOF では `0.990`, `0.932`、実環境では `0.991`, `0.919` だった。
- YOLO26m も高 confidence の検出に絞られており、CEPDOF では `0.950`, `0.941`、実環境では `0.912`, `0.706` だった。
- YOLOv8m は低 confidence の候補も含めて多く検出し、実環境では `0.293` の検出も含まれた。

### 領域一致度

- YOLO26m は axis-aligned bbox を出力するため、魚眼画像で斜め方向に写る人物に対して bbox が広がりやすい。
- RAPiD は rotated bbox を出力するため、魚眼画像上の人物姿勢に合わせた領域表現ができる。
- Ground Truth が無いため IoU による定量評価は未実施だが、魚眼画像向けの領域表現としては RAPiD の方が適している。

### 推論速度

- YOLO 系は全体的に RAPiD より高速だった。
- 平均推論時間は YOLOv8m が `5.2 ms`、YOLO11m が `6.2 ms`、YOLO26m が `6.4 ms`、RAPiD が `47.2 ms` だった。
- RAPiD は YOLO 系より約 7.4 - 9.1 倍遅い。
- YOLO 系の世代差は小さいが、この2枚では YOLOv8m が最速かつ検出数が最多だった。

### モデルサイズ

- YOLO26m は `42.2 MB`。
- YOLO11m は `38.8 MB`。
- YOLOv8m は `49.7 MB`。
- RAPiD は `235.0 MB`。
- 配置先のストレージやメモリ制約を重視する場合は YOLO 系の方が扱いやすい。

## 結論

魚眼画像の人物領域をより正確に表現したい場合は RAPiD を優先する。理由は、魚眼画像に合わせた rotated bbox を出力でき、CEPDOF と実環境の両方で高 confidence の人物検出ができたためである。

一方で、実環境での速度、モデルサイズ、実装の扱いやすさを重視する場合は YOLO 系が有利である。今回の2枚では YOLOv8m が最速かつ検出数も最も多かった。YOLO26m と YOLO11m は検出数が RAPiD と同じで、confidence も高く、RAPiD より大幅に高速だった。

実運用候補としては、rotated bbox が必須なら RAPiD、axis-aligned bbox で十分なら YOLOv8m または YOLO26m を優先する。最終判断には Ground Truth bbox を用意し、IoU、Precision、Recall を追加で評価する必要がある。
