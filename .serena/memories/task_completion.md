# Task completion checklist
- Ensure dependencies installed via `pip install -e .` before running scripts.
- For pose-estimation changes, validate by running `python src/hpe/run_sapiens.py` (or variant) on sample images and confirm outputs written to expected directories.
- No automated tests currently; manual verification required.
- Document any assumptions about data locations (e.g., `data/`, `output/from_video/`) in PR/notes.