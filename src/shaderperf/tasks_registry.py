"""Load evaluation tasks; only ``agent_file`` is agent-visible material.

JSON stores ``never_shown`` as a list. The loader returns it as a set of
repository-relative paths, including each reference and any injection notes.
This is a manifest for callers to enforce, not a filesystem access sandbox.
"""

from __future__ import annotations

import json
import math
from pathlib import Path


def load_registry(root: str | Path | None = None) -> list[dict]:
    """Read tasks/registry.json relative to *root*, independent of the cwd.

    Invalid records raise ValueError; missing/unreadable registry files retain
    their normal OSError. Returned records are fresh on every call.
    """
    root = Path(root).resolve() if root is not None else Path(__file__).resolve().parents[2]
    records = json.loads((root / "tasks" / "registry.json").read_text(encoding="utf-8"))
    if not isinstance(records, list) or not records:
        raise ValueError("task registry must be a nonempty list")

    def require_file(value: object) -> Path:
        if not isinstance(value, str) or not value or Path(value).is_absolute():
            raise ValueError(f"expected a repository-relative file path: {value!r}")
        path = (root / value).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            raise ValueError(f"task file missing or outside repository: {value!r}")
        return path

    ids = set()
    hidden_paths = set()
    agent_paths = set()
    source_keys = {
        "title", "author", "url", "license", "license_evidence",
        "popularity", "contamination_flag",
    }
    for record in records:
        if not isinstance(record, dict):
            raise ValueError("each task must be an object")
        task_id = record.get("task_id")
        if not isinstance(task_id, str) or not task_id or task_id in ids:
            raise ValueError(f"invalid or duplicate task_id: {task_id!r}")
        ids.add(task_id)
        if record.get("stratum") not in ("natural", "injected"):
            raise ValueError(f"{task_id}: invalid stratum")
        batch = record.get("batch")
        if type(batch) is not int or batch <= 0:
            raise ValueError(f"{task_id}: batch must be a positive integer")
        duty = record.get("duty")
        if type(duty) not in (int, float) or not 0 < duty <= 1:
            raise ValueError(f"{task_id}: duty must be a number in (0, 1]")
        source = record.get("source")
        if not isinstance(source, dict) or not source_keys.issubset(source):
            raise ValueError(f"{task_id}: incomplete source metadata")
        times = record.get("visible_times")
        if not isinstance(times, list) or not times or any(
            type(t) not in (int, float) or not math.isfinite(t) for t in times
        ):
            raise ValueError(f"{task_id}: visible_times must be finite numbers")
        if "headroom" not in record or "sealed_task" not in record:
            raise ValueError(f"{task_id}: missing measurement/commitment fields")

        agent = require_file(record.get("agent_file"))
        reference = require_file(record.get("reference_file"))
        if agent == reference:
            raise ValueError(f"{task_id}: agent and reference must be different files")
        hidden = record.get("never_shown")
        if not isinstance(hidden, list) or not hidden:
            raise ValueError(f"{task_id}: never_shown must list the reference and notes")
        paths = {require_file(value) for value in hidden}
        if record["reference_file"] not in hidden:
            raise ValueError(f"{task_id}: reference must be never_shown")
        if record["stratum"] == "injected":
            notes = record.get("injection_notes")
            require_file(notes)
            if notes not in hidden:
                raise ValueError(f"{task_id}: injection notes must be never_shown")
        record["never_shown"] = set(hidden)
        hidden_paths.update(paths)
        agent_paths.add(agent)
    if agent_paths & hidden_paths:
        raise ValueError("agent-visible files overlap never_shown material")
    return records
