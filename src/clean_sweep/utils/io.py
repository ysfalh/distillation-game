from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_json(data: Any, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def write_jsonl(rows: list[dict[str, Any]], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def write_markdown_examples(rows: list[dict[str, Any]], path: str | Path, *, title: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"# {title}", ""]
    for row in rows:
        lines.extend(
            [
                f"## {row['example_id']}",
                "",
                "Prompt:",
                "",
                row.get("prompt", "").strip(),
                "",
                "Trace:",
                "",
                row.get("trace", "").strip(),
                "",
                "Answer-Forced Final Answer:",
                "",
                (row.get("af_final_answer_only") or "").strip(),
                "",
                f"Raw extracted answer: `{row.get('raw_extracted_answer')}`",
                f"AF extracted answer: `{row.get('af_extracted_answer')}`",
                f"Raw correct: `{row.get('raw_correct')}`",
                f"AF correct: `{row.get('af_correct')}`",
                f"Extracted answer: `{row.get('extracted_answer')}`",
                f"Correct: `{row.get('correct')}`",
                "",
                "---",
                "",
            ]
        )
    path.write_text("\n".join(lines))
