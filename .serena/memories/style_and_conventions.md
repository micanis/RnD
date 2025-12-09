# Style and conventions
- Python with type hints used throughout; dataclasses for structured paths (`BasePaths`).
- Imports prefer absolute via `src` on sys.path fallback; paths resolved relative to project root.
- CLI interactivity via `questionary` prompts (Japanese messages).
- JSON outputs use UTF-8 and `ensure_ascii=False` for keypoint dumps.
- No explicit formatting/lint config found; follow standard PEP8 and keep comments concise. Default encoding UTF-8 per pyproject metadata.