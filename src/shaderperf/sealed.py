"""Deterministic sealed-sample commitments for evaluation tasks."""

from __future__ import annotations

import hashlib
import json
import math
import secrets
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


DEFAULT_N = 12
DEFAULT_T_RANGE = (0.0, 10.0)


def _repo_root(root: str | Path | None) -> Path:
    if root is not None:
        return Path(root)
    return Path(__file__).parent.parent.parent


def _sealed_paths(root: str | Path | None) -> tuple[Path, Path]:
    sealed_dir = _repo_root(root) / "sealed"
    return sealed_dir / "commitments.json", sealed_dir / "secrets.json"


def _parameters(
    n: int, t_range: tuple[float, float]
) -> tuple[int, float, float]:
    if isinstance(n, bool) or not isinstance(n, int) or n <= 0:
        raise ValueError("n must be a positive integer")
    try:
        t0, t1 = (float(value) for value in t_range)
    except (TypeError, ValueError) as exc:
        raise ValueError("t_range must contain exactly two numbers") from exc
    if not math.isfinite(t0) or not math.isfinite(t1) or t1 <= t0:
        raise ValueError("t_range must be finite and increasing")
    return n, t0, t1


def sealed_times(
    secret_seed: str,
    task_id: str,
    n: int = DEFAULT_N,
    t_range: tuple[float, float] = DEFAULT_T_RANGE,
) -> list[float]:
    """Return one deterministic pseudorandom time from each equal stratum."""

    n, t0, t1 = _parameters(n, t_range)
    digest = hashlib.sha256(f"{task_id}|{secret_seed}".encode()).digest()
    rng = np.random.default_rng(int.from_bytes(digest[:8], "big"))
    stratum_width = (t1 - t0) / n
    times = [
        t0 + (index + float(rng.random())) * stratum_width
        for index in range(n)
    ]
    return sorted(round(value, 6) for value in times)


def commitment_hash(
    task_id: str,
    secret_seed: str,
    n: int,
    t_range: tuple[float, float],
) -> str:
    """Hash the canonical public commitment payload for one task."""

    n, t0, t1 = _parameters(n, t_range)
    canonical = f"{task_id}|{secret_seed}|{n}|{t0:.6f}|{t1:.6f}"
    return hashlib.sha256(canonical.encode()).hexdigest()


def _read_mapping(path: Path) -> dict:
    if not path.exists():
        return {}
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object in {path}")
    return value


def _write_mapping(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def load_commitment(task_id: str, root: str | Path | None = None) -> dict:
    """Load a task's public commitment record without reading its secret."""

    commitments_path, _secrets_path = _sealed_paths(root)
    commitments = _read_mapping(commitments_path)
    try:
        record = commitments[task_id]
    except KeyError as exc:
        raise KeyError(f"no commitment for task {task_id!r}") from exc
    if not isinstance(record, dict):
        raise ValueError(f"invalid commitment record for task {task_id!r}")
    return dict(record)


def _record_parameters(task_id: str, record: dict) -> tuple[int, tuple[float, float]]:
    try:
        n = record["n"]
        t_range = record["t_range"]
        commitment = record["commitment"]
    except KeyError as exc:
        raise ValueError(f"incomplete commitment record for task {task_id!r}") from exc
    n, t0, t1 = _parameters(n, t_range)
    if not isinstance(commitment, str):
        raise ValueError(f"invalid commitment record for task {task_id!r}")
    return n, (t0, t1)


def _load_secret(task_id: str, root: str | Path | None) -> str:
    _commitments_path, secrets_path = _sealed_paths(root)
    stored_secrets = _read_mapping(secrets_path)
    try:
        secret_seed = stored_secrets[task_id]
    except KeyError as exc:
        raise KeyError(f"no local secret for task {task_id!r}") from exc
    if not isinstance(secret_seed, str):
        raise ValueError(f"invalid local secret for task {task_id!r}")
    return secret_seed


def _verified_parameters(
    task_id: str, secret_seed: str, record: dict
) -> tuple[int, tuple[float, float]]:
    n, t_range = _record_parameters(task_id, record)
    expected = commitment_hash(task_id, secret_seed, n, t_range)
    if not secrets.compare_digest(record["commitment"], expected):
        raise ValueError(f"commitment mismatch for task {task_id!r}")
    return n, t_range


def commit_task(
    task_id: str,
    n: int = DEFAULT_N,
    t_range: tuple[float, float] = DEFAULT_T_RANGE,
    secret_seed: str | None = None,
    root: str | Path | None = None,
    force: bool = False,
) -> dict:
    """Create and persist a task's public commitment and local secret."""

    n, t0, t1 = _parameters(n, t_range)
    commitments_path, secrets_path = _sealed_paths(root)
    commitments = _read_mapping(commitments_path)
    if task_id in commitments and not force:
        raise ValueError(f"commitment already exists for task {task_id!r}")

    stored_secrets = _read_mapping(secrets_path)
    secret_seed = secrets.token_hex(16) if secret_seed is None else secret_seed
    if not isinstance(secret_seed, str):
        raise ValueError("secret_seed must be a string")

    record = {
        "commitment": commitment_hash(task_id, secret_seed, n, (t0, t1)),
        "n": n,
        "t_range": [t0, t1],
        "committed_utc": datetime.now(timezone.utc)
        .isoformat(timespec="seconds")
        .replace("+00:00", "Z"),
        "revealed_seed": None,
    }
    stored_secrets[task_id] = secret_seed
    commitments[task_id] = record
    _write_mapping(secrets_path, stored_secrets)
    _write_mapping(commitments_path, commitments)
    return dict(record)


def load_sealed_times(
    task_id: str, root: str | Path | None = None
) -> list[float]:
    """Verify a local secret against its commitment and regenerate times."""

    record = load_commitment(task_id, root)
    secret_seed = _load_secret(task_id, root)
    n, t_range = _verified_parameters(task_id, secret_seed, record)
    return sealed_times(secret_seed, task_id, n, t_range)


def reveal_task(task_id: str, root: str | Path | None = None) -> dict:
    """Verify and publish a task's secret, returning the reveal payload."""

    commitments_path, _secrets_path = _sealed_paths(root)
    commitments = _read_mapping(commitments_path)
    record = load_commitment(task_id, root)
    secret_seed = _load_secret(task_id, root)
    n, t_range = _verified_parameters(task_id, secret_seed, record)
    times = sealed_times(secret_seed, task_id, n, t_range)

    record["revealed_seed"] = secret_seed
    commitments[task_id] = record
    _write_mapping(commitments_path, commitments)
    return {
        "seed": secret_seed,
        "times": times,
        "commitment": record["commitment"],
    }


def verify_commitment(
    task_id: str, secret_seed: str, root: str | Path | None = None
) -> bool:
    """Check a supplied seed against a task's public commitment."""

    record = load_commitment(task_id, root)
    n, t_range = _record_parameters(task_id, record)
    actual = commitment_hash(task_id, secret_seed, n, t_range)
    return secrets.compare_digest(record["commitment"], actual)
