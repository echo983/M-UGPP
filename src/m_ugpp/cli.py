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
        default=20,
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


def run(goal: str, parser: argparse.ArgumentParser, args: argparse.Namespace) -> dict:
    discoverer, planner, worker, evaluator = build_default_roles()
    config = UGPPConfig(
        max_discovery_rounds=args.max_discovery_rounds,
        max_execution_rounds=args.max_execution_rounds,
        max_tasks_per_round=args.max_tasks_per_round,
        min_mts_confidence=args.min_mts_confidence,
    )

    mts = _build_initial_mts(goal, args.truth, args.need)
    engine = UGPPEngine(
        discoverer=discoverer,
        planner=planner,
        worker=worker,
        evaluator=evaluator,
        config=config,
    )

    final_report = engine.run(goal, initial_mts=mts)
    result = {
        "status": final_report.status.value,
        "message": final_report.message,
        "phase": final_report.state.phase.value,
        "completed": list(final_report.state.completed_nodes),
        "mts_truths": len(final_report.state.mts.truths),
        "mts_confidence": final_report.state.mts.confidence,
    }

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
