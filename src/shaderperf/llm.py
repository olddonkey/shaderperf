"""LLM clients and durable spend accounting for agent trajectories."""

from __future__ import annotations

import json
import math
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
PRICES_PATH = REPO_ROOT / "config" / "agent_prices.json"
LEDGER_PATH = REPO_ROOT / "results" / "agent" / "ledger.json"
BUDGET_USD = 100.0


class OpenAIChat:
    """Small adapter around the OpenAI chat completions API."""

    def __init__(self, model: str, api_key: str, base_url: str | None = None):
        from openai import OpenAI

        kwargs: dict[str, Any] = {"api_key": api_key}
        if base_url is not None:
            kwargs["base_url"] = base_url
        self.model = model
        self.client = OpenAI(**kwargs)

    def complete(
        self, messages: list[dict], max_output_tokens: int
    ) -> dict[str, str | int]:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_completion_tokens=max_output_tokens,
        )
        usage = response.usage
        return {
            "text": response.choices[0].message.content or "",
            "input_tokens": int(usage.prompt_tokens) if usage else 0,
            "output_tokens": int(usage.completion_tokens) if usage else 0,
        }


class MockChat:
    """Deterministic scripted chat client used by tests and the mock CLI."""

    def __init__(self, responses: list[str]):
        self.responses = list(responses)

    def complete(
        self, messages: list[dict], max_output_tokens: int
    ) -> dict[str, str | int]:
        del messages, max_output_tokens
        if not self.responses:
            raise RuntimeError("MockChat has no scripted responses remaining")
        text = self.responses.pop(0)
        if not isinstance(text, str):
            raise TypeError("MockChat responses must be strings")
        fake_tokens = len(text) // 4
        return {
            "text": text,
            "input_tokens": fake_tokens,
            "output_tokens": fake_tokens,
        }


def build_codex_argv(
    *,
    model: str,
    effort: str,
    workspace_dir: str | Path,
    prompt: str,
    codex_bin: str = "codex",
    extra_config: list[str] | None = None,
    thread_id: str | None = None,
) -> list[str]:
    """Build a first-turn or resume ``codex exec`` argument vector."""

    argv = [codex_bin, "exec"]
    if thread_id is not None:
        argv.extend(["resume", thread_id])
    argv.extend(["--json", "--skip-git-repo-check"])
    if thread_id is None:
        argv.extend(["-s", "read-only", "-C", str(workspace_dir), "-m", model])
    argv.extend(["-c", f'model_reasoning_effort="{effort}"'])
    if thread_id is not None:
        argv.extend(["-c", 'sandbox_mode="read-only"'])
    for value in extra_config or []:
        argv.extend(["-c", value])
    argv.extend(["-c", "mcp_servers={}", prompt])
    return argv


def parse_codex_jsonl(text: str) -> dict[str, Any]:
    """Parse the useful fields from ``codex exec --json`` JSONL output."""

    thread_id: str | None = None
    final_text = ""
    warnings: list[str] = []
    usage: dict[str, Any] = {}
    turn_completed = False

    for line in text.splitlines():
        try:
            event = json.loads(line)
        except (json.JSONDecodeError, TypeError):
            continue
        if not isinstance(event, dict):
            continue
        event_type = event.get("type")
        if event_type == "thread.started":
            candidate = event.get("thread_id")
            if isinstance(candidate, str) and candidate:
                thread_id = candidate
        elif event_type == "item.completed":
            item = event.get("item")
            if not isinstance(item, dict):
                continue
            item_type = item.get("type")
            if item_type == "agent_message":
                candidate = item.get("text")
                if isinstance(candidate, str):
                    final_text = candidate
            elif item_type == "error":
                candidate = item.get("message", item.get("text"))
                if isinstance(candidate, str):
                    warnings.append(candidate)
                else:
                    warnings.append(json.dumps(item, sort_keys=True))
        elif event_type == "turn.completed":
            turn_completed = True
            candidate = event.get("usage")
            usage = candidate if isinstance(candidate, dict) else {}

    def token_count(name: str) -> int:
        try:
            return int(usage.get(name, 0))
        except (TypeError, ValueError):
            return 0

    result: dict[str, Any] = {
        "text": final_text,
        "input_tokens": token_count("input_tokens"),
        "output_tokens": token_count("output_tokens"),
    }
    if thread_id is not None:
        result["thread_id"] = thread_id
    if "cached_input_tokens" in usage:
        result["cached_input_tokens"] = token_count("cached_input_tokens")
    if "reasoning_output_tokens" in usage:
        result["reasoning_output_tokens"] = token_count(
            "reasoning_output_tokens"
        )
    if not turn_completed:
        warnings.append("missing turn.completed event; usage defaults to zero")
    result["warnings"] = warnings
    return result


def _subprocess_output(value: str | bytes | None) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value


def _output_tail(stdout: str | bytes | None, stderr: str | bytes | None) -> str:
    combined = "\n".join(
        part for part in (_subprocess_output(stdout), _subprocess_output(stderr)) if part
    )
    return combined[-800:]


class CodexChat:
    """Stateful adapter around subscription-backed ``codex exec`` threads."""

    def __init__(
        self,
        model: str,
        effort: str,
        workspace_dir: str | Path,
        codex_bin: str = "codex",
        extra_config: list[str] | None = None,
        timeout_seconds: float = 600.0,
    ):
        workspace = Path(workspace_dir).expanduser().resolve()
        if workspace == REPO_ROOT.resolve():
            raise ValueError("Codex workspace must be a scratch directory, not the repo")
        workspace.mkdir(parents=True, exist_ok=True)
        self.model = model
        self.effort = effort
        self.workspace_dir = workspace
        self.codex_bin = codex_bin
        self.extra_config = list(extra_config or [])
        self.timeout_seconds = float(timeout_seconds)
        self.thread_id: str | None = None
        self.warnings: list[str] = []
        self.max_output_tokens_requests: list[int] = []
        self.argv_flags_history: list[list[str]] = []
        self._message_count: int | None = None
        self._run = subprocess.run

    def complete(
        self, messages: list[dict], max_output_tokens: int
    ) -> dict[str, Any]:
        if self._message_count is not None and len(messages) < self._message_count:
            raise RuntimeError(
                "CodexChat message history shrank between complete() calls"
            )

        first_turn = self._message_count is None
        if first_turn:
            if (
                len(messages) < 2
                or messages[0].get("role") != "system"
                or messages[1].get("role") != "user"
            ):
                raise ValueError(
                    "first CodexChat call requires a system message followed by "
                    "a user message"
                )
            prompt = f"{messages[0].get('content', '')}\n\n{messages[1].get('content', '')}"
            resume_thread_id = None
        else:
            if self.thread_id is None:
                raise RuntimeError("CodexChat thread_id is missing for resume")
            if not messages:
                raise ValueError("CodexChat requires at least one message")
            prompt = str(messages[-1].get("content", ""))
            resume_thread_id = self.thread_id

        requested_tokens = int(max_output_tokens)
        self.max_output_tokens_requests.append(requested_tokens)
        argv = build_codex_argv(
            model=self.model,
            effort=self.effort,
            workspace_dir=self.workspace_dir,
            prompt=prompt,
            codex_bin=self.codex_bin,
            extra_config=self.extra_config,
            thread_id=resume_thread_id,
        )
        self.argv_flags_history.append(argv[1:-1])
        try:
            completed = self._run(
                argv,
                capture_output=True,
                text=True,
                timeout=self.timeout_seconds,
                cwd=self.workspace_dir,
            )
        except subprocess.TimeoutExpired as exc:
            tail = _output_tail(exc.stdout, exc.stderr)
            detail = f"\n{tail}" if tail else ""
            raise RuntimeError(
                f"Codex CLI timed out after {self.timeout_seconds:g} seconds{detail}"
            ) from exc

        if completed.returncode != 0:
            tail = _output_tail(completed.stdout, completed.stderr)
            detail = f"\n{tail}" if tail else ""
            raise RuntimeError(
                f"Codex CLI exited with status {completed.returncode}{detail}"
            )

        parsed = parse_codex_jsonl(completed.stdout)
        if first_turn:
            parsed_thread_id = parsed.get("thread_id")
            if not isinstance(parsed_thread_id, str) or not parsed_thread_id:
                raise RuntimeError(
                    "Codex CLI output did not include a thread.started thread_id"
                )
            self.thread_id = parsed_thread_id
        parsed["thread_id"] = self.thread_id
        latest_warnings = list(parsed.get("warnings", []))
        self.warnings.extend(latest_warnings)
        parsed["warnings"] = latest_warnings
        parsed["requested_max_output_tokens"] = requested_tokens
        self._message_count = len(messages)
        return parsed


def resolve_api_key(repo_root: str | Path | None = None) -> str:
    """Resolve the API key without ever displaying any part of it."""

    environment_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if environment_key:
        return environment_key

    root = Path(repo_root) if repo_root is not None else REPO_ROOT
    key_path = root / ".secrets" / "openai_key"
    if key_path.is_file():
        file_key = key_path.read_text(encoding="utf-8").strip()
        if file_key:
            return file_key

    raise RuntimeError(
        "OpenAI API key not found. Set OPENAI_API_KEY or put the key as a "
        f"single line in {key_path}."
    )


def load_prices(path: str | Path = PRICES_PATH) -> dict[str, dict[str, float]]:
    price_path = Path(path)
    try:
        raw = json.loads(price_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise RuntimeError(f"agent price file is missing: {price_path}") from exc
    if not isinstance(raw, dict) or not raw:
        raise ValueError(
            f"agent price file must contain a non-empty object: {price_path}"
        )

    prices: dict[str, dict[str, float]] = {}
    for model, entry in raw.items():
        if not isinstance(model, str) or not isinstance(entry, dict):
            raise ValueError(f"invalid price entry for {model!r}")
        try:
            input_price = float(entry["input_per_mtok"])
            output_price = float(entry["output_per_mtok"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"invalid price entry for {model!r}") from exc
        if (
            not math.isfinite(input_price)
            or not math.isfinite(output_price)
            or input_price < 0.0
            or output_price < 0.0
        ):
            raise ValueError(
                f"prices must be finite and non-negative for {model!r}"
            )
        prices[model] = {
            "input_per_mtok": input_price,
            "output_per_mtok": output_price,
        }
    return prices


def empty_ledger() -> dict[str, Any]:
    return {
        "spent_usd": 0.0,
        "input_tokens": 0,
        "output_tokens": 0,
        "runs": [],
    }


def load_ledger(path: str | Path = LEDGER_PATH) -> dict[str, Any]:
    ledger_path = Path(path)
    if not ledger_path.exists():
        return empty_ledger()
    ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    if not isinstance(ledger, dict) or not isinstance(ledger.get("runs"), list):
        raise ValueError(f"invalid agent ledger: {ledger_path}")
    try:
        spent = float(ledger["spent_usd"])
        input_tokens = int(ledger["input_tokens"])
        output_tokens = int(ledger["output_tokens"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"invalid agent ledger totals: {ledger_path}") from exc
    if (
        not math.isfinite(spent)
        or spent < 0.0
        or input_tokens < 0
        or output_tokens < 0
    ):
        raise ValueError(f"agent ledger totals must be non-negative: {ledger_path}")
    ledger["spent_usd"] = spent
    ledger["input_tokens"] = input_tokens
    ledger["output_tokens"] = output_tokens
    return ledger


def add_run_usage(
    ledger: dict[str, Any],
    run: dict[str, Any] | None = None,
    *,
    input_tokens: int | None = None,
    output_tokens: int | None = None,
    cost_usd: float | None = None,
    **metadata: Any,
) -> dict[str, Any]:
    """Append one usage record and update ledger totals in place."""

    record = dict(run or {})
    if input_tokens is not None:
        record["input_tokens"] = input_tokens
    if output_tokens is not None:
        record["output_tokens"] = output_tokens
    if cost_usd is not None:
        record["cost_usd"] = cost_usd
    record.update(metadata)

    try:
        run_input = int(record["input_tokens"])
        run_output = int(record["output_tokens"])
        run_cost = float(record.get("cost_usd", record.get("spent_usd")))
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            "run usage requires input_tokens, output_tokens, and cost_usd"
        ) from exc
    if (
        run_input < 0
        or run_output < 0
        or not math.isfinite(run_cost)
        or run_cost < 0.0
    ):
        raise ValueError("run usage values must be non-negative")

    normalized = record | {
        "input_tokens": run_input,
        "output_tokens": run_output,
        "cost_usd": run_cost,
    }
    normalized.pop("spent_usd", None)
    ledger.setdefault("runs", []).append(normalized)
    ledger["input_tokens"] = int(ledger.get("input_tokens", 0)) + run_input
    ledger["output_tokens"] = int(ledger.get("output_tokens", 0)) + run_output
    ledger["spent_usd"] = float(ledger.get("spent_usd", 0.0)) + run_cost
    return ledger


def _atomic_write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, path)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise


def save_ledger(
    ledger: dict[str, Any], path: str | Path = LEDGER_PATH
) -> None:
    _atomic_write_json(Path(path), ledger)


def calculate_cost_usd(
    prices: dict[str, dict[str, float]],
    model: str,
    input_tokens: int,
    output_tokens: int,
) -> float:
    try:
        price = prices[model]
    except KeyError as exc:
        raise RuntimeError(f"no configured price for model {model!r}") from exc
    return (
        int(input_tokens) * float(price["input_per_mtok"])
        + int(output_tokens) * float(price["output_per_mtok"])
    ) / 1_000_000.0


def check_run_allowed(
    ledger: dict[str, Any],
    prices: dict[str, dict[str, float]],
    model: str,
    *,
    allow_unpriced: bool = False,
    budget_usd: float = BUDGET_USD,
) -> None:
    """Refuse a real run when its global dollar budget is unenforceable."""

    spent = float(ledger.get("spent_usd", 0.0))
    if not math.isfinite(spent) or spent < 0.0:
        raise RuntimeError("agent ledger spent_usd must be finite and non-negative")
    if not math.isfinite(budget_usd) or budget_usd <= 0.0:
        raise ValueError("budget_usd must be finite and positive")
    if spent >= budget_usd:
        raise RuntimeError(
            f"global agent spend cap reached: ${spent:.6f} spent, "
            f"cap is ${budget_usd:.2f}"
        )
    if model not in prices:
        raise RuntimeError(f"no configured price for model {model!r}")
    selected = prices[model]
    model_unpriced = (
        float(selected["input_per_mtok"]) == 0.0
        and float(selected["output_per_mtok"]) == 0.0
    )
    all_unpriced = all(
        float(entry["input_per_mtok"]) == 0.0
        and float(entry["output_per_mtok"]) == 0.0
        for entry in prices.values()
    )
    if (model_unpriced or all_unpriced) and not allow_unpriced:
        raise RuntimeError(
            "dollar budget cannot be enforced because model prices are all "
            "zero; fill in config/agent_prices.json or pass --allow-unpriced"
        )
