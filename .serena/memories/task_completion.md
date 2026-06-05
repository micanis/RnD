# Task Completion

- For Python/workspace changes, run from repo root: `devbox run lint`.
- For dependency metadata moves/edits, run from `workspace/`: `uv --cache-dir .uv-cache lock --check`.
- For Devbox package changes, run from repo root: `devbox install`; network may be required to update/install packages.
- For Typst availability, run from repo root: `devbox run -- typst --version`.
- Before finalizing, check `git status --short` and distinguish task changes from pre-existing dirty files.