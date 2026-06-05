# Tech Stack

- Python project: Python >=3.11, uv-managed, project files in `workspace/`.
- Python source layout: `workspace/src/core` and `workspace/src/tools`.
- Python dependencies include CV/ML stack: OpenCV, NumPy, SciPy, torch/torchvision, ultralytics, zarr, typer, questionary.
- Root dev environment: Devbox with uv, ruff, basedpyright, bun, typst.
- Slide project: Bun-based project in `docs/slide/` with its own `package.json` and `bun.lock`.
- Typst documents belong in `docs/typst/`.