# Suggested Commands

- Enter repo dev environment from repo root: `devbox shell`.
- Lint Python workspace from repo root: `devbox run lint`.
- Format Python workspace from repo root: `devbox run fmt`.
- Check Python lock from workspace: `uv --cache-dir .uv-cache lock --check`.
- Work with Python dependencies from workspace: `cd workspace && uv sync`.
- Run ad hoc Typst commands from repo root: `devbox run -- typst --version` or `devbox run -- typst compile docs/typst/<file>.typ`.
- Slide project commands should be run from `docs/slide/` unless package scripts say otherwise.