# Conventions

- Keep repo root as orchestrator for shared tools and assets; do not move Python packaging files back to root.
- Keep Python-specific files under `workspace/`.
- Keep shared datasets and model artifacts at root-level `data/` and `models/`.
- Python code that needs shared assets should resolve the repo root from file location or `PROJECT_ROOT`, not from cwd.
- Keep independent docs toolchains under `docs/<tool-or-medium>/`, e.g. `docs/slide/`, `docs/typst/`.
- Avoid broad refactors in `tmp/`; tracked `tmp/` files may be in user-managed dirty state.