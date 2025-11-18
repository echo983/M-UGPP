"""Command-line entrypoint for running the UGPP demo pipeline."""
from __future__ import annotations

import argparse
import json
from typing import Iterable, Sequence

from .engine import UGPPEngine
from .roles import build_default_roles, seed_mts
from .types import MTS, Priority, Truth, TruthNeed, UGPPConfig


def _parse_truths(raw_values: Sequence[str]) -> Iterable[Truth]:
    from datetime import datetime
    from .types import Source

    truths: list[Truth] = []
    for raw in raw_values:
        try:
            truth_id, statement, confidence_str = raw.split("|", 2)
            confidence = float(confidence_str)
        except ValueError as exc:  # pragma: no cover - defensive parsing
            raise argparse.ArgumentTypeError(
                "Truth must be formatted as 'id|statement|confidence'"
            ) from exc

        truths.append(
            Truth(
                id=truth_id,
                statement=statement,
                sources=frozenset({Source(type="manual", uri="cli", reliability=0.9)}),
                confidence=confidence,
                timestamp=datetime.utcnow(),
                verifiable=True,
            )
        )
    return truths


def _parse_needs(raw_values: Sequence[str]) -> Iterable[TruthNeed]:
    needs: list[TruthNeed] = []
    for raw in raw_values:
        try:
            need_id, question, confidence_str, priority_str, reason = raw.split("|", 4)
            confidence = float(confidence_str)
            priority = Priority[priority_str.upper()]
        except (ValueError, KeyError) as exc:  # pragma: no cover - defensive parsing
            raise argparse.ArgumentTypeError(
                "Need must be 'id|question|confidence|priority|reason'"
            ) from exc
        needs.append(
            TruthNeed(
                id=need_id,
                question=question,
                required_confidence=confidence,
                priority=priority,
                reason=reason,
            )
        )
    return needs


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the Minimal UGPP demo pipeline")
    parser.add_argument("goal", help="Overall goal to accomplish")
    parser.add_argument(
        "--max-discovery-rounds",
        type=int,
        default=3,
        help="Maximum number of discovery rounds before failing",
    )
    parser.add_argument(
        "--max-execution-rounds",
        type=int,
        default=10,
        help="Maximum number of execution rounds before timing out",
    )
    parser.add_argument(
        "--max-tasks-per-round",
        type=int,
        default=32,
        help="Number of tasks the worker can execute per round",
    )
    parser.add_argument(
        "--min-mts-confidence",
        type=float,
        default=0.7,
        help="Minimum MTS confidence required before planning",
    )
    parser.add_argument(
        "--truth",
        action="append",
        default=[],
        metavar="ID|STATEMENT|CONFIDENCE",
        help="Seed the MTS with a truth (can be repeated)",
    )
    parser.add_argument(
        "--need",
        action="append",
        default=[],
        metavar="ID|QUESTION|CONFIDENCE|PRIORITY|REASON",
        help="Add a missing truth need (can be repeated)",
    )
    parser.add_argument(
        "--dump-json",
        action="store_true",
        help="Output the final report as JSON for automation",
    )
    parser.add_argument(
        "--task",
        action="append",
        default=[],
        metavar="EXECUTOR:PAYLOAD",
        help="Add an executable task (executor is 'bash' or 'python')",
    )
    parser.add_argument(
        "--bash",
        action="append",
        default=[],
        metavar="CMD",
        help="Shortcut for adding a bash task (appended after --task entries)",
    )
    parser.add_argument(
        "--python",
        action="append",
        default=[],
        metavar="CODE",
        help="Shortcut for adding a python task executed with `python -c`",
    )
    parser.add_argument(
        "--workdir",
        default=".",
        help="Working directory used for all executable tasks",
    )
    parser.add_argument(
        "--task-timeout",
        type=float,
        default=90.0,
        help="Maximum seconds allowed for each task before timing out",
    )
    parser.add_argument(
        "--log-file",
        default=None,
        help="Optional path to write JSONL logs for the run",
    )
    parser.add_argument(
        "--no-gpt",
        action="store_true",
        help="Use local rule-based roles instead of GPT-powered roles",
    )
    parser.add_argument(
        "--planner-model",
        default=None,
        help="Override GPT planner model (default gpt-5.1)",
    )
    parser.add_argument(
        "--worker-model",
        default=None,
        help="Override GPT worker model (default gpt-5-mini)",
    )
    parser.add_argument(
        "--discoverer-model",
        default=None,
        help="Override GPT discoverer model (default gpt-5-nano)",
    )
    parser.add_argument(
        "--evaluator-model",
        default=None,
        help="Override GPT evaluator model (default gpt-5-mini)",
    )
    return parser


def _build_initial_mts(goal: str, truths: Sequence[str], needs: Sequence[str]) -> MTS:
    mts = seed_mts(goal)
    parsed_truths = list(_parse_truths(truths)) if truths else []
    parsed_needs = list(_parse_needs(needs)) if needs else []

    if parsed_truths:
        mts.truths |= set(parsed_truths)
    if parsed_needs:
        mts.missing |= set(parsed_needs)

    mts.check_sufficiency()
    return mts


def _parse_tasks(args: argparse.Namespace, parser: argparse.ArgumentParser) -> list[tuple[str, str]]:
    tasks: list[tuple[str, str]] = []
    for raw in args.task:
        if ":" not in raw:
            parser.error("Task must be formatted as 'bash:<command>' or 'python:<code>'")
        executor, payload = raw.split(":", 1)
        executor = executor.strip().lower()
        if executor not in {"bash", "python"}:
            parser.error("Task executor must be 'bash' or 'python'")
        tasks.append((executor, payload))

    for cmd in args.bash:
        tasks.append(("bash", cmd))
    for code in args.python:
        tasks.append(("python", code))

    return tasks


def _model_overrides(args: argparse.Namespace) -> dict[str, str]:
    models: dict[str, str] = {}
    if args.planner_model:
        models["planner"] = args.planner_model
    if args.worker_model:
        models["worker"] = args.worker_model
    if args.discoverer_model:
        models["discoverer"] = args.discoverer_model
    if args.evaluator_model:
        models["evaluator"] = args.evaluator_model
    return models


def run(goal: str, parser: argparse.ArgumentParser, args: argparse.Namespace) -> dict:
    tasks = _parse_tasks(args, parser)
    use_gpt = not args.no_gpt
    model_overrides = _model_overrides(args)
    discoverer, planner, worker, evaluator = build_default_roles(
        tasks=tasks,
        workdir=args.workdir,
        use_gpt=use_gpt,
        models=model_overrides or None,
    )
    config = UGPPConfig(
        max_discovery_rounds=args.max_discovery_rounds,
        max_execution_rounds=args.max_execution_rounds,
        max_tasks_per_round=args.max_tasks_per_round,
        min_mts_confidence=args.min_mts_confidence,
        task_timeout=args.task_timeout,
    )
    log_file = args.log_file
    log_handle = open(log_file, "a", encoding="utf-8") if log_file else None

    def logger(message: str, details: dict | None = None) -> None:
        if not log_handle:
            return
        record = {"message": message, "details": details or {}}
        log_handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        log_handle.flush()

    mts = _build_initial_mts(goal, args.truth, args.need)
    engine = UGPPEngine(
        discoverer=discoverer,
        planner=planner,
        worker=worker,
        evaluator=evaluator,
        config=config,
    )

    final_report = engine.run(goal, initial_mts=mts, log=logger)
    result = {
        "status": final_report.status.value,
        "message": final_report.message,
        "phase": final_report.state.phase.value,
        "completed": list(final_report.state.completed_nodes),
        "mts_truths": len(final_report.state.mts.truths),
        "mts_confidence": final_report.state.mts.confidence,
        "tasks_requested": len(tasks),
        "gpt_enabled": use_gpt,
    }
    if log_handle:
        log_handle.close()

    payload = json.dumps(result, indent=None if args.dump_json else 2)

    if args.dump_json:
        print(payload)
        return result

    parser.exit(status=0, message=payload + "\n")

    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(args=argv)
    run(args.goal, parser, args)
    return 0


if __name__ == "__main__":
    main()
