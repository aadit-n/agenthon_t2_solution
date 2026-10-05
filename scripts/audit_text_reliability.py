"""Offline audit of supplied corpora and prompt target semantics, without any House calls."""

from __future__ import annotations

import argparse
import json
import pathlib
import tomllib
from collections import Counter

from t2solution import legacy_text
from t2solution.text import output_budget
from t2solution.text_context import read_documents, task_context
from t2solution.unit import load_unit


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=pathlib.Path, required=True)
    parser.add_argument("--out", type=pathlib.Path)
    args = parser.parse_args()
    rows, exclusions, changes = [], Counter(), 0
    for root in sorted((args.repo / "units").iterdir()):
        if not (root / "forecast_spec.json").is_file():
            continue
        card = tomllib.loads((root / "card.toml").read_text(encoding="utf-8"))
        asof = str(card["provenance"]["data_cutoff"])
        unit = load_unit(root, root / "text", asof)
        audit = {}
        docs = read_documents(unit.text_dir, asof=asof, audit=audit)
        old_docs = legacy_text._read_documents(unit.text_dir)
        ctx = task_context(unit, asof)
        selected = [doc["doc_id"] for doc in docs]
        changed = selected != [doc["doc_id"] for doc in old_docs]
        changes += changed
        exclusions.update(audit["document_exclusions"])
        rows.append({"unit_id": unit.unit_id, "asof": asof, **audit, "selected_doc_ids": selected,
                     "legacy_selection_changed": changed, "asset_context": ctx["asset_context"],
                     "extraction_tokens": output_budget("extraction", len(unit.assets)),
                     "synthesis_tokens": output_budget("synthesis", len(unit.assets))})
    summary = {"units": len(rows), "selected_documents": sum(row["documents_selected"] for row in rows),
               "units_with_no_documents": sum(row["documents_selected"] == 0 for row in rows),
               "units_with_selection_changes": changes, "exclusions": dict(exclusions),
               "max_selected_documents": max((row["documents_selected"] for row in rows), default=0),
               "max_extraction_tokens": max((row["extraction_tokens"] for row in rows), default=0),
               "max_synthesis_tokens": max((row["synthesis_tokens"] for row in rows), default=0),
               "house_calls": 0, "predictive_score": None}
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps({"summary": summary, "units": rows}, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
