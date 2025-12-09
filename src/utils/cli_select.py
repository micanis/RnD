from pathlib import Path
from typing import Callable, Iterable, Sequence

import questionary


def _select(prompt: str, options: Iterable[Path]) -> Path:
    options = sorted(options)
    if not options:
        raise RuntimeError(f"{prompt} の候補がありません。")
    selected = questionary.select(
        prompt,
        choices=[questionary.Choice(p.name, value=p) for p in options],
    ).ask()
    if selected is None:
        raise RuntimeError("選択がキャンセルされました。")
    return selected


def select_hierarchy(
    root: Path,
    steps: Sequence[tuple[str, Callable[[Path], Iterable[Path]]]],
) -> Path:
    """
    階層を1つずつ questionary.select でたどる汎用ヘルパー。

    steps: [(prompt, selector)] で selector は current_path を受け取り Path の反復を返す。
    """
    current = root
    for prompt, selector in steps:
        current = _select(prompt, selector(current))
    return current


def select_path(
    message: str,
    base_dir: Path | None = None,
    *,
    kind: str = "dir",
    extensions: Iterable[str] | None = None,
    choices: Iterable[Path] | None = None,
    allow_manual: bool = True,
    create: bool = False,
) -> Path | None:
    """
    ファイル/ディレクトリを questionary で選択する簡易ヘルパー。
    - choices があればそれを使う。なければ base_dir 直下を列挙。
    - allow_manual=True なら手入力も許可。
    - kind == "dir" かつ create=True なら存在しない場合に生成。
    """
    base_dir = base_dir or Path.cwd()
    manual_token = "__manual__"
    normalized_exts = {
        ext if ext.startswith(".") else f".{ext}"
        for ext in (ext.lower() for ext in extensions or [])
    }

    if create and kind == "dir":
        base_dir.mkdir(parents=True, exist_ok=True)

    if choices is not None:
        candidates = sorted(Path(c) for c in choices)
    elif base_dir.exists():
        if kind == "dir":
            candidates = sorted(p for p in base_dir.iterdir() if p.is_dir())
        else:
            candidates = sorted(
                p
                for p in base_dir.iterdir()
                if p.is_file() and (not normalized_exts or p.suffix.lower() in normalized_exts)
            )
    else:
        candidates = []

    question_choices = [questionary.Choice(str(p), value=p) for p in candidates]
    if allow_manual:
        question_choices.append(questionary.Choice("別のパスを指定する", value=manual_token))

    if not question_choices:
        return None

    selected = questionary.select(message, choices=question_choices).ask()
    if selected is None:
        return None

    if selected == manual_token:
        manual_input = questionary.path(
            "パスを入力してください:",
            default=str(base_dir),
            only_directories=kind == "dir",
        ).ask()
        if not manual_input:
            return None
        selected_path = Path(manual_input).expanduser()
    else:
        selected_path = Path(selected)

    if not selected_path.exists():
        if kind == "dir" and create:
            selected_path.mkdir(parents=True, exist_ok=True)
        else:
            raise FileNotFoundError(f"指定されたパスが存在しません: {selected_path}")
    return selected_path
