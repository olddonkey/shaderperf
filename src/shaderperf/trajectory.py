"""Run one feedback Condition-A shader-optimization trajectory."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import time
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import wgpu

from .harness import GpuContext, ShaderTimer
from .llm import (
    BUDGET_USD,
    LEDGER_PATH,
    MockChat,
    OpenAIChat,
    _atomic_write_json,
    add_run_usage,
    calculate_cost_usd,
    check_run_allowed,
    load_ledger,
    load_prices,
    resolve_api_key,
    save_ledger,
)
from .protocol import interleaved_compare
from .verifier import evaluate_sample


REPO_ROOT = Path(__file__).resolve().parents[2]
CALIBRATION_REPORT_PATH = (
    REPO_ROOT / "results" / "calibration" / "calibration_report.json"
)

SYSTEM_PROMPT = """You are optimizing a WGSL fragment shader for GPU time on a fixed device.
Preserve the rendered image: perceptual equivalence is checked, and hidden held-out checks also exist but are not shown to you. Every optimization reply must contain the complete modified shader in one ```wgsl fenced block. If a reply contains more than one such block, only the last block is evaluated. Do not change the vertex stage or the Params uniform interface. Do not use external resources. After at least one shader submission has compiled successfully, you may reply with the literal token FINAL by itself to end the trajectory; that reply does not need a shader block."""

WGSL_BLOCK = re.compile(
    r"```[ \t]*wgsl[ \t]*\r?\n(.*?)```", re.IGNORECASE | re.DOTALL
)


class ChatClient(Protocol):
    def complete(
        self, messages: list[dict], max_output_tokens: int
    ) -> dict[str, str | int]: ...


class Evaluator(Protocol):
    def evaluate(self, source: str) -> dict[str, Any]: ...


def load_tau_flip(path: str | Path = CALIBRATION_REPORT_PATH) -> float:
    report_path = Path(path)
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise RuntimeError(
            f"FLIP calibration report is missing: {report_path}"
        ) from exc
    try:
        threshold = float(report["tau_FLIP"])
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError(
            f"FLIP calibration report has no valid tau_FLIP: {report_path}"
        ) from exc
    if not math.isfinite(threshold) or threshold < 0.0:
        raise RuntimeError(
            f"FLIP calibration report has no valid tau_FLIP: {report_path}"
        )
    return threshold


def extract_last_wgsl(text: str) -> str | None:
    """Return the last WGSL fenced block, or ``None`` when absent."""

    matches = WGSL_BLOCK.findall(text)
    return matches[-1].strip() if matches else None


class TrajectoryEvaluator:
    """GPU-backed evaluator comparing every attempt with the original task."""

    def __init__(
        self,
        context: GpuContext,
        task_path: str | Path,
        *,
        visible_times: list[float],
        tau_flip: float,
        pairs: int,
        batch: int,
        duty: float,
        seed: int,
        width: int = 1024,
        height: int = 1024,
    ):
        if not visible_times:
            raise ValueError("at least one visible time is required")
        self.context = context
        self.task_path = Path(task_path)
        self.visible_times = [float(value) for value in visible_times]
        self.tau_flip = float(tau_flip)
        self.pairs = int(pairs)
        self.batch = int(batch)
        self.duty = float(duty)
        self.width = int(width)
        self.height = int(height)
        self.rng = np.random.default_rng(seed)

        shared = {
            "width": self.width,
            "height": self.height,
            "context": self.context,
        }
        self.task_timer = ShaderTimer(self.task_path, **shared)
        self.reference_float_timer = ShaderTimer(
            self.task_path,
            **shared,
            texture_format=wgpu.TextureFormat.rgba32float,
        )

    def evaluate(self, source: str) -> dict[str, Any]:
        shared = {
            "width": self.width,
            "height": self.height,
            "context": self.context,
            "source_override": source,
        }
        try:
            submission_timer = ShaderTimer(self.task_path, **shared)
            submission_float_timer = ShaderTimer(
                self.task_path,
                **shared,
                texture_format=wgpu.TextureFormat.rgba32float,
            )
        except Exception as exc:
            return {
                "compile_ok": False,
                "compile_error": str(exc),
                "numerical": None,
                "visible_flips": [],
                "speedup_ci": None,
            }

        visible_flips: list[dict[str, Any]] = []
        numerical: dict[str, Any] | None = None
        for time_value in self.visible_times:
            sample = evaluate_sample(
                self.reference_float_timer, submission_float_timer, time_value
            )
            if numerical is None:
                numerical = dict(sample["numerical_test"])
            flip_value = float(sample["flip_mean"])
            visible_flips.append(
                {
                    "time": float(time_value),
                    "flip_mean": flip_value,
                    "passed": bool(flip_value <= self.tau_flip),
                }
            )

        comparison = interleaved_compare(
            self.task_timer,
            submission_timer,
            pairs=self.pairs,
            batch=self.batch,
            warmup=3,
            rng=self.rng,
            time_value=self.visible_times[0],
            duty=self.duty,
        )
        return {
            "compile_ok": True,
            "compile_error": None,
            "numerical": numerical,
            "visible_flips": visible_flips,
            "speedup_ci": dict(comparison["speedup_ci"]),
        }


def _first_user_message(task_source: str, tau_flip: float) -> str:
    return f"""Optimize this complete WGSL shader:

```wgsl
{task_source}
```

After each attempt you will receive: the compilation outcome and compiler diagnostic if any; the numerical range/finiteness result at the first visible time; FLIP at every visible time with pass/fail against tau_FLIP={tau_flip:.9f}; the total speedup point estimate and confidence interval against the original shader; and the remaining attempt and token budgets."""


def _missing_block_feedback(attempts_remaining: int, tokens_remaining: int) -> str:
    return (
        "Attempt feedback (Condition A)\n"
        "Submission: INVALID - no ```wgsl fenced block was found.\n"
        "This reply consumed one attempt.\n"
        f"Attempts remaining: {attempts_remaining}\n"
        f"Token budget remaining: {tokens_remaining}"
    )


def format_feedback(
    evaluation: dict[str, Any],
    *,
    tau_flip: float,
    attempts_remaining: int,
    tokens_remaining: int,
) -> str:
    """Format only the baseline feedback contract exposed to the agent."""

    lines = ["Attempt feedback (Condition A)"]
    if not evaluation["compile_ok"]:
        lines.append("Compilation: FAIL")
        lines.append(
            f"Compiler error: {evaluation.get('compile_error') or 'unknown error'}"
        )
    else:
        lines.append("Compilation: PASS")
        numerical = evaluation.get("numerical") or {}
        lines.append(
            "Numerical check: "
            f"{'PASS' if numerical.get('passed') else 'FAIL'}; "
            f"has_nan={numerical.get('has_nan')}; "
            f"has_inf={numerical.get('has_inf')}; "
            f"min={numerical.get('min')}; max={numerical.get('max')}; "
            f"in_range={numerical.get('in_range')}"
        )
        lines.append(f"Visible FLIP (tau_FLIP={tau_flip:.9f}):")
        for sample in evaluation.get("visible_flips", []):
            verdict = "PASS" if sample["passed"] else "FAIL"
            lines.append(
                f"- time={float(sample['time']):g}: "
                f"FLIP={float(sample['flip_mean']):.9f} {verdict}"
            )
        speedup = evaluation.get("speedup_ci")
        if speedup is None:
            lines.append("Speedup vs original: unavailable")
        else:
            lines.append(
                "Speedup vs original: "
                f"{float(speedup['speedup_point']):.6f}x; "
                f"CI [{float(speedup['ci_low']):.6f}, "
                f"{float(speedup['ci_high']):.6f}]"
            )
    lines.append(f"Attempts remaining: {attempts_remaining}")
    lines.append(f"Token budget remaining: {tokens_remaining}")
    return "\n".join(lines)


def _validate_label(label: str) -> None:
    if not label or Path(label).name != label or label in {".", ".."}:
        raise ValueError(
            "label must be a non-empty filename without path separators"
        )


def _utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def run_trajectory(
    *,
    chat: ChatClient,
    evaluator: Evaluator,
    task_path: str | Path,
    task_source: str,
    label: str,
    out_dir: str | Path,
    condition: str = "A",
    max_attempts: int = 8,
    max_total_tokens: int = 200_000,
    max_output_tokens: int = 8_000,
    visible_times: list[float] | tuple[float, ...] = (1.0, 3.7, 9.2),
    pairs: int = 12,
    batch: int = 32,
    duty: float = 0.3,
    seed: int = 0,
    model: str = "gpt-5.6-sol",
    base_url: str | None = None,
    tau_flip: float,
    env_meta: dict[str, Any] | None = None,
    real_run: bool = False,
    prices: dict[str, dict[str, float]] | None = None,
    ledger: dict[str, Any] | None = None,
    ledger_path: str | Path = LEDGER_PATH,
    allow_unpriced: bool = False,
    budget_usd: float = BUDGET_USD,
) -> dict[str, Any]:
    """Run the mock-testable loop and return its persisted trajectory log."""

    if condition != "A":
        raise ValueError("only condition A is implemented")
    _validate_label(label)
    if max_attempts < 0 or max_total_tokens < 0 or max_output_tokens <= 0:
        raise ValueError(
            "attempt/token limits must be non-negative and output positive"
        )
    visible = [float(value) for value in visible_times]
    if not visible:
        raise ValueError("at least one visible time is required")
    if real_run:
        if prices is None or ledger is None:
            raise ValueError("real runs require prices and a ledger")
        check_run_allowed(
            ledger,
            prices,
            model,
            allow_unpriced=allow_unpriced,
            budget_usd=budget_usd,
        )

    output_directory = Path(out_dir)
    partial_path = output_directory / f"{label}.partial.json"
    final_path = output_directory / f"{label}.json"
    task_path = Path(task_path)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": _first_user_message(task_source, tau_flip)},
    ]
    trajectory: dict[str, Any] = {
        "schema_version": 1,
        "label": label,
        "started_utc": _utc_now(),
        "config": {
            "task_path": str(task_path),
            "task_sha256": hashlib.sha256(task_source.encode("utf-8")).hexdigest(),
            "condition": condition,
            "max_attempts": max_attempts,
            "max_total_tokens": max_total_tokens,
            "max_output_tokens": max_output_tokens,
            "visible_times": visible,
            "pairs": pairs,
            "batch": batch,
            "duty": duty,
            "seed": seed,
            "model": model,
            "base_url": base_url,
            "tau_flip": float(tau_flip),
            "mock": not real_run,
            "allow_unpriced": bool(allow_unpriced),
            "budget_usd": float(budget_usd),
        },
        "env_meta": dict(env_meta or {}),
        "messages": messages,
        "attempts": [],
        "usage": {
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
            "cost_usd": 0.0,
        },
        "stop_reason": None,
    }

    valid_submission_seen = False
    if max_attempts == 0:
        trajectory["stop_reason"] = "max_attempts"
    elif max_total_tokens == 0:
        trajectory["stop_reason"] = "token_budget"

    while trajectory["stop_reason"] is None:
        attempt_started = time.perf_counter()
        attempt_number = len(trajectory["attempts"]) + 1
        tokens_before = int(trajectory["usage"]["total_tokens"])
        remaining_before = max_total_tokens - tokens_before
        if remaining_before <= 0:
            trajectory["stop_reason"] = "token_budget"
            break

        response = chat.complete(
            messages,
            max_output_tokens=min(max_output_tokens, remaining_before),
        )
        text = str(response["text"])
        input_tokens = int(response["input_tokens"])
        output_tokens = int(response["output_tokens"])
        if input_tokens < 0 or output_tokens < 0:
            raise ValueError("LLM token usage must be non-negative")
        call_cost = (
            calculate_cost_usd(prices, model, input_tokens, output_tokens)
            if real_run and prices is not None
            else 0.0
        )
        trajectory["usage"]["input_tokens"] += input_tokens
        trajectory["usage"]["output_tokens"] += output_tokens
        trajectory["usage"]["total_tokens"] += input_tokens + output_tokens
        trajectory["usage"]["cost_usd"] += call_cost
        messages.append({"role": "assistant", "content": text})

        if real_run and ledger is not None:
            add_run_usage(
                ledger,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cost_usd=call_cost,
                label=label,
                attempt=attempt_number,
                model=model,
                utc_time=_utc_now(),
            )
            save_ledger(ledger, ledger_path)

        usage_record = {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
            "cost_usd": call_cost,
            "cumulative_input_tokens": trajectory["usage"]["input_tokens"],
            "cumulative_output_tokens": trajectory["usage"]["output_tokens"],
            "cumulative_total_tokens": trajectory["usage"]["total_tokens"],
            "cumulative_cost_usd": trajectory["usage"]["cost_usd"],
        }

        if text.strip() == "FINAL" and valid_submission_seen:
            attempt = {
                "attempt": attempt_number,
                "final_requested": True,
                "submission_sha256": None,
                "submission_source": None,
                "compile_ok": None,
                "compile_error": None,
                "numerical": None,
                "visible_flips": [],
                "speedup_ci": None,
                "wall_seconds": time.perf_counter() - attempt_started,
                "usage": usage_record,
            }
            trajectory["attempts"].append(attempt)
            trajectory["stop_reason"] = "final"
            _atomic_write_json(partial_path, trajectory)
            break

        submission = extract_last_wgsl(text)
        attempts_remaining = max_attempts - attempt_number
        tokens_remaining = max(
            0, max_total_tokens - int(trajectory["usage"]["total_tokens"])
        )
        if submission is None:
            attempt = {
                "attempt": attempt_number,
                "final_requested": False,
                "submission_sha256": None,
                "submission_source": None,
                "compile_ok": False,
                "compile_error": "no WGSL fenced block",
                "numerical": None,
                "visible_flips": [],
                "speedup_ci": None,
                "wall_seconds": time.perf_counter() - attempt_started,
                "usage": usage_record,
            }
            feedback = _missing_block_feedback(
                attempts_remaining, tokens_remaining
            )
        else:
            evaluation = evaluator.evaluate(submission)
            attempt = {
                "attempt": attempt_number,
                "final_requested": False,
                "submission_sha256": hashlib.sha256(
                    submission.encode("utf-8")
                ).hexdigest(),
                "submission_source": submission,
                "compile_ok": bool(evaluation["compile_ok"]),
                "compile_error": evaluation.get("compile_error"),
                "numerical": evaluation.get("numerical"),
                "visible_flips": list(evaluation.get("visible_flips", [])),
                "speedup_ci": evaluation.get("speedup_ci"),
                "wall_seconds": time.perf_counter() - attempt_started,
                "usage": usage_record,
            }
            if attempt["compile_ok"]:
                valid_submission_seen = True
            feedback = format_feedback(
                evaluation,
                tau_flip=tau_flip,
                attempts_remaining=attempts_remaining,
                tokens_remaining=tokens_remaining,
            )

        trajectory["attempts"].append(attempt)
        messages.append({"role": "user", "content": feedback})

        global_cap_reached = bool(
            real_run
            and ledger is not None
            and float(ledger["spent_usd"]) >= budget_usd
        )
        if global_cap_reached:
            trajectory["stop_reason"] = "dollar_budget"
        elif int(trajectory["usage"]["total_tokens"]) >= max_total_tokens:
            trajectory["stop_reason"] = "token_budget"
        elif attempt_number >= max_attempts:
            trajectory["stop_reason"] = "max_attempts"

        _atomic_write_json(partial_path, trajectory)

    trajectory["completed_utc"] = _utc_now()
    _atomic_write_json(final_path, trajectory)
    return trajectory


def _parse_visible_times(raw: str) -> list[float]:
    try:
        values = [float(value.strip()) for value in raw.split(",") if value.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "visible times must be a comma-separated list of numbers"
        ) from exc
    if not values:
        raise argparse.ArgumentTypeError("at least one visible time is required")
    return values


def _positive_int(raw: str) -> int:
    value = int(raw)
    if value <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task", type=Path, metavar="TASK.wgsl")
    parser.add_argument("--label", required=True, metavar="L")
    parser.add_argument("--condition", choices=["A"], default="A")
    parser.add_argument("--max-attempts", type=_positive_int, default=8)
    parser.add_argument("--max-total-tokens", type=_positive_int, default=200_000)
    parser.add_argument("--max-output-tokens", type=_positive_int, default=8_000)
    parser.add_argument(
        "--visible-times", type=_parse_visible_times, default=[1.0, 3.7, 9.2]
    )
    parser.add_argument("--pairs", type=_positive_int, default=12)
    parser.add_argument("--batch", type=_positive_int, default=32)
    parser.add_argument("--duty", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--model", default="gpt-5.6-sol")
    parser.add_argument("--base-url")
    parser.add_argument("--mock", type=Path, metavar="SCRIPT.json")
    parser.add_argument("--allow-unpriced", action="store_true")
    parser.add_argument(
        "--out", type=Path, default=Path("results/agent/trajectories")
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    try:
        _validate_label(args.label)
    except ValueError as exc:
        parser.error(str(exc))
    if not 0.0 < args.duty <= 1.0:
        parser.error("--duty must be in the interval (0, 1]")

    task_source = args.task.read_text(encoding="utf-8")
    tau_flip = load_tau_flip()

    prices = None
    ledger = None
    real_run = args.mock is None
    if args.mock is not None:
        scripted = json.loads(args.mock.read_text(encoding="utf-8"))
        if not isinstance(scripted, list) or not all(
            isinstance(item, str) for item in scripted
        ):
            parser.error("--mock SCRIPT.json must contain a list of strings")
        chat: ChatClient = MockChat(scripted)
    else:
        prices = load_prices()
        ledger = load_ledger()
        check_run_allowed(
            ledger,
            prices,
            args.model,
            allow_unpriced=args.allow_unpriced,
        )
        chat = OpenAIChat(
            args.model, resolve_api_key(), base_url=args.base_url
        )

    context = GpuContext()
    evaluator = TrajectoryEvaluator(
        context,
        args.task,
        visible_times=args.visible_times,
        tau_flip=tau_flip,
        pairs=args.pairs,
        batch=args.batch,
        duty=args.duty,
        seed=args.seed,
    )
    result = run_trajectory(
        chat=chat,
        evaluator=evaluator,
        task_path=args.task,
        task_source=task_source,
        label=args.label,
        out_dir=args.out,
        condition=args.condition,
        max_attempts=args.max_attempts,
        max_total_tokens=args.max_total_tokens,
        max_output_tokens=args.max_output_tokens,
        visible_times=args.visible_times,
        pairs=args.pairs,
        batch=args.batch,
        duty=args.duty,
        seed=args.seed,
        model=args.model,
        base_url=args.base_url,
        tau_flip=tau_flip,
        env_meta=context.meta(),
        real_run=real_run,
        prices=prices,
        ledger=ledger,
        allow_unpriced=args.allow_unpriced,
    )
    print(f"wrote {args.out / f'{args.label}.json'}")
    print(f"stop reason: {result['stop_reason']}")


if __name__ == "__main__":
    main()
