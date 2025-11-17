"""Concrete role implementations for a runnable UGPP demo pipeline."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Iterable, Set

from .dag import DAG
from .interfaces import (
    Discoverer,
    DiscoveryRequest,
    DiscoveryResult,
    EvalReport,
    Evaluator,
    Planner,
    PlanningRequest,
    PlanningResult,
    Worker,
    WorkerJob,
    WorkerResult,
)
from .types import MTS, Priority, Source, TaskNode, Truth, TruthNeed


def _default_sources() -> Set[Source]:
    return {Source(type="memory", uri="local", reliability=1.0)}


@dataclass
class SimpleDiscoverer(Discoverer):
    """A deterministic discoverer that fabricates truths from the goal."""

    confidence: float = 0.85

    def discover(self, request: DiscoveryRequest) -> DiscoveryResult:  # noqa: D401
        """Produce a new truth describing progress toward the goal."""

        truth_id = f"truth-{request.round}"
        statement = f"Resolved need for '{request.goal}' round {request.round}"
        truth = Truth(
            id=truth_id,
            statement=statement,
            sources=frozenset(_default_sources()),
            confidence=self.confidence,
            timestamp=datetime.utcnow(),
            verifiable=True,
        )

        return DiscoveryResult(
            new_truths={truth},
            confidence=self.confidence,
            sources_used=set(_default_sources()),
        )


@dataclass
class SimplePlanner(Planner):
    """Plan execution as a single atomic task based on the goal."""

    def _build_task(self, goal: str) -> TaskNode:
        return TaskNode(
            id="task-complete-goal",
            type="atomic",
            preconditions=[],
            postconditions=[],
            estimated_cost=1.0,
            metadata={"goal": goal},
        )

    def plan(self, request: PlanningRequest) -> PlanningResult:  # noqa: D401
        """Return a minimal DAG that attempts the requested goal."""

        node = self._build_task(request.goal)
        dag = DAG(nodes={node.id: node}, edges=set())
        estimated_rounds = max(1, len(dag.nodes))
        return PlanningResult(dag=dag, estimated_rounds=estimated_rounds, confidence=0.8)


@dataclass
class SimpleWorker(Worker):
    """Execute atomic tasks by returning a success payload."""

    def execute(self, job: WorkerJob) -> WorkerResult:  # noqa: D401
        """Mark the job as completed and return a synthetic artifact."""

        node = job.node
        node.status = "success"
        artifacts: Dict[str, Any] = {
            "message": f"Completed task '{node.id}' for goal '{node.metadata.get('goal', '')}'",
            "timestamp": datetime.utcnow().isoformat(),
        }
        return WorkerResult(
            node_id=node.id,
            status="success",
            artifacts=artifacts,
            execution_time=0.1,
        )


@dataclass
class SimpleEvaluator(Evaluator):
    """Evaluate progress based on completed DAG nodes and MTS confidence."""

    min_confidence: float = 0.7

    def _needs_met(self, mts: MTS) -> bool:
        if not mts.missing:
            return True

        missing_priorities = {need.priority for need in mts.missing}
        # If critical needs remain, we cannot succeed.
        if Priority.CRITICAL in missing_priorities:
            return False

        return mts.confidence >= self.min_confidence

    def evaluate(self, state: "UGPPState") -> EvalReport:  # noqa: D401
        """Return a simple report describing overall progress."""

        dag = state.dag
        all_completed = bool(dag) and len(state.completed_nodes) == len(dag.nodes)
        mts = state.mts
        mts.check_sufficiency(self.min_confidence)

        if all_completed and self._needs_met(mts):
            overall = "ok"
        elif mts.sufficient and not dag:
            overall = "early_success"
        else:
            overall = "ok" if mts.sufficient else "need_fix"

        details: Dict[str, Dict[str, Any]] = {
            "mts": {
                "confidence": mts.confidence,
                "truths": len(mts.truths),
                "missing": [need.id for need in mts.missing],
            }
        }

        if dag:
            details["dag"] = {
                "completed": list(state.completed_nodes),
                "remaining": [
                    node_id for node_id in dag.nodes.keys() if node_id not in state.completed_nodes
                ],
            }

        recommendations: list[str] = []
        if overall == "need_fix":
            recommendations.append("Add more truths or relax confidence requirements.")

        return EvalReport(overall=overall, details=details, recommendations=recommendations)


def seed_mts(goal: str, confidence: float = 0.6) -> MTS:
    """Generate a starting MTS with a single medium-priority need."""

    need = TruthNeed(
        id="need-1",
        question=f"What is required to achieve '{goal}'?",
        required_confidence=confidence,
        priority=Priority.MEDIUM,
        reason="Baseline need for demo pipeline",
    )

    return MTS(truths=set(), missing={need})


def build_default_roles() -> tuple[Discoverer, Planner, Worker, Evaluator]:
    """Helper to instantiate the default role implementations."""

    return (SimpleDiscoverer(), SimplePlanner(), SimpleWorker(), SimpleEvaluator())


def update_mts_from_truths(mts: MTS, new_truths: Iterable[Truth]) -> MTS:
    """Apply discovered truths and recompute sufficiency."""

    mts.truths |= set(new_truths)
    mts.check_sufficiency()
    return mts
