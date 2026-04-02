from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_no_internal_imports_from_deprecated_casual_module() -> None:
    forbidden = "from neraium_core.casual import"
    offenders: list[str] = []

    for path in REPO_ROOT.rglob("*.py"):
        rel = path.relative_to(REPO_ROOT).as_posix()
        if rel in {"neraium_core/casual.py", "tests/test_no_casual_imports.py"}:
            continue
        text = path.read_text(encoding="utf-8")
        if forbidden in text:
            offenders.append(rel)

    assert offenders == [], f"Found deprecated casual.py imports: {offenders}"
