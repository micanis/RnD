# Suggested commands
- Install deps: `python -m pip install -e .` (uses pyproject.toml dependencies: torch, torchvision, ultralytics, opencv-contrib-python, huggingface-hub, questionary, tqdm).
- Run pose estimation: `python src/hpe/run_sapiens.py` (select input/output dirs via prompts unless configured in RUN_CONFIG).
- Calibrate fisheye camera: `python src/tools/calibrate_fisheye.py` (expects frames under `output/from_video/calibration/left`).
- No tests or lint commands defined; add/execute ad-hoc scripts as needed.