# Core

- Repo is a multi-workspace R&D repo, not a single Python-only project.
- Root owns shared development environment and repo-level assets: `devbox.json`, `devbox.lock`, `.envrc`, `README.md`, `data/`, `models/`, `docs/`.
- Python project lives under `workspace/`: `workspace/pyproject.toml`, `workspace/uv.lock`, `workspace/.python-version`, `workspace/src/`.
- `data/` and `models/` intentionally remain root-level shared assets; Python code should resolve repo root, not assume cwd is repo root or workspace root.
- Docs contain independent tool projects: slides under `docs/slide/`, Typst documents under `docs/typst/`.

Read `mem:tech_stack` for tools and package managers. Read `mem:task_completion` for verification commands.