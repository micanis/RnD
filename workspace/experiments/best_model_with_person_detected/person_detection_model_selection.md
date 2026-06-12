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
- YOLO: `yolo26m.pt`, `yolo11m.pt`, `models/yolov8m.pt`
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
| YOLO26m | `cepdof_lunch1_0500.jpg` | 2 | 再測定待ち | 42.2 MB | `outputs/yolo/yolo26m/cepdof_lunch1_0500_yolo26m.jpg` |
| YOLO26m | `real_test2_000.jpg` | 2 | 再測定待ち | 42.2 MB | `outputs/yolo/yolo26m/real_test2_000_yolo26m.jpg` |
| YOLO11m | `cepdof_lunch1_0500.jpg` | 再測定待ち | 再測定待ち | 再測定待ち | `outputs/yolo/yolo11m/cepdof_lunch1_0500_yolo11m.jpg` |
| YOLO11m | `real_test2_000.jpg` | 再測定待ち | 再測定待ち | 再測定待ち | `outputs/yolo/yolo11m/real_test2_000_yolo11m.jpg` |
| YOLOv8m | `cepdof_lunch1_0500.jpg` | 再測定待ち | 再測定待ち | 49.7 MB | `outputs/yolo/yolov8m/cepdof_lunch1_0500_yolov8m.jpg` |
| YOLOv8m | `real_test2_000.jpg` | 再測定待ち | 再測定待ち | 49.7 MB | `outputs/yolo/yolov8m/real_test2_000_yolov8m.jpg` |
| RAPiD | `cepdof_lunch1_0500.jpg` | 2 | 229.6 ms | 235.0 MB | `outputs/rapid/cepdof_lunch1_0500_rapid.jpg` |
| RAPiD | `real_test2_000.jpg` | 2 | 46.3 ms | 235.0 MB | `outputs/rapid/real_test2_000_rapid.jpg` |

| モデル | 平均検出数 | 平均推論時間 | モデルサイズ |
| --- | ---: | ---: | ---: |
| YOLO26m | 2.0 | 再測定待ち | 42.2 MB |
| YOLO11m | 再測定待ち | 再測定待ち | 再測定待ち |
| YOLOv8m | 再測定待ち | 再測定待ち | 49.7 MB |
| RAPiD | 2.0 | 138.0 ms | 235.0 MB |

## 評価

### 検出精度

今回の2枚の評価では Ground Truth bbox を用意していないため、厳密な Precision / Recall / IoU は算出していない。検出数と confidence、および描画結果の目視確認をもとに比較する。

- YOLO26m は初回測定では CEPDOF と実環境の両方で 2 件を検出した。
- YOLO11m と YOLOv8m はウォームアップ除外後の再測定で確認する。
- RAPiD は CEPDOF と実環境の両方で 2 件を検出した。
- RAPiD の検出 confidence は高く、CEPDOF では `0.990`, `0.932`、実環境では `0.991`, `0.919` だった。
- YOLO26m も高 confidence の検出に絞られており、CEPDOF では `0.950`, `0.941`、実環境では `0.912`, `0.706` だった。

### 領域一致度

- YOLO26m は axis-aligned bbox を出力するため、魚眼画像で斜め方向に写る人物に対して bbox が広がりやすい。
- RAPiD は rotated bbox を出力するため、魚眼画像上の人物姿勢に合わせた領域表現ができる。
- Ground Truth が無いため IoU による定量評価は未実施だが、魚眼画像向けの領域表現としては RAPiD の方が適している。

### 推論速度

- 旧測定ではモデルロード後の初回推論が含まれていたため、速度比較から除外する。
- 修正版の実験コードでは `warmup_runs` 後に `measure_runs` 回推論し、平均値、中央値、最小値、各回の測定値を JSON に保存する。
- YOLO は `yolo26m.pt`, `yolo11m.pt`, `models/yolov8m.pt` を同一条件で測定し、バージョン差異を比較する。

### モデルサイズ

- YOLO26m は `42.2 MB`。
- RAPiD は `235.0 MB`。
- 配置先のストレージやメモリ制約を重視する場合は YOLO26m の方が扱いやすい。

## 結論

現時点では、魚眼画像の人物領域を安定して扱う候補として RAPiD を優先する。理由は、魚眼画像に合わせた rotated bbox を出力でき、CEPDOF と実環境の両方で高 confidence の人物検出ができたためである。

一方で、RAPiD はモデルサイズが大きい。YOLO 系はモデルサイズが小さく、最新世代では検出 confidence も高いため、軽量な配置や高速処理を優先する用途では YOLO26m / YOLO11m / YOLOv8m のバージョン差異を確認してから最終候補を選ぶ。最終判断には Ground Truth bbox を用意し、IoU、Precision、Recall を追加で評価する必要がある。
