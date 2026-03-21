from __future__ import annotations

import re
from typing import Any


def _md_table(headers: list[str], rows: list[list[str]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def build_results_report(rows: list[dict[str, Any]], *, dataset: str, teacher: str, student: str) -> str:
    teacher_rows = [r for r in rows if str(r.get("eval_model", "")).startswith("teacher_")]
    student_rows = [r for r in rows if str(r.get("eval_model", "")).startswith("student_")]

    out: list[str] = ["# Results", ""]
    teacher_test_acc: dict[str, str] = {}
    for row in teacher_rows:
        eval_model = str(row["eval_model"])
        if not eval_model.endswith("_test"):
            continue
        teacher_key = eval_model[: -len("_test")]
        teacher_test_acc[teacher_key] = f"{float(row['accuracy']):.4f}"

    def _student_col_label(row: dict[str, Any]) -> str:
        eval_model = str(row["eval_model"])
        notes = str(row.get("notes", ""))
        m = re.search(r"beta_s=([^,\s]+)", notes)
        beta = m.group(1) if m else None
        return f"{eval_model} (beta_s={beta})" if beta is not None else eval_model

    train_sources = sorted(set(teacher_test_acc) | {str(r["train_source"]) for r in student_rows})
    student_labels = sorted({_student_col_label(r) for r in student_rows})
    student_lookup = {
        (str(r["train_source"]), _student_col_label(r)): r
        for r in student_rows
    }

    if train_sources:
        out.extend(["## Comparison Matrix", ""])
        headers = ["Teacher"] + ["Teacher Accuracy"] + student_labels
        matrix_rows: list[list[str]] = []
        for source in train_sources:
            row = [source, teacher_test_acc.get(source, "")]
            for label in student_labels:
                match = student_lookup.get((source, label))
                if match is None:
                    row.append("")
                    continue
                cell = f"{float(match['accuracy']):.4f}"
                notes = str(match.get("notes", "")).strip()
                if notes:
                    notes_no_beta = re.sub(r"^beta_s=[^,]+,?\s*", "", notes)
                    if notes_no_beta:
                        cell += f"<br><sub>{notes_no_beta}</sub>"
                row.append(cell)
            matrix_rows.append(row)
        out.append(_md_table(headers, matrix_rows))
        out.append("")

    out.extend(
        [
            "Model Context:",
            f"- Dataset: `{dataset}`",
            f"- Teacher model: `{teacher}`",
            f"- Student model: `{student}`",
            "",
        ]
    )
    return "\n".join(out)
