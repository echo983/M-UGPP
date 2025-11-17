"""UGPP engine coordinating discovery, planning, execution, and evaluation."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

from .interfaces import (
    Discoverer,
    DiscoveryRequest,
    Evaluator,
    Planner,
    PlanningRequest,
    Worker,
    WorkerJob,
)
from .types import FinalReport, MTS, Phase, Status, UGPPConfig, UGPPState


@dataclass
class UGPPEngine:
    discoverer: Discoverer
    planner: Planner
    worker: Worker
    evaluator: Evaluator
    config: UGPPConfig = field(default_factory=UGPPConfig)

    def run(
        self,
        goal: str,
        initial_mts: Optional[MTS] = None,
        log: Callable[[str, dict | None], None] | None = None,
    ) -> FinalReport:
        mts = initial_mts or MTS()
        state = UGPPState(phase=Phase.DISCOVERY, round=1, goal=goal, mts=mts)
        cfg = self.config

        def _log(message: str, details: dict | None = None) -> None:
            if log:
                log(message, details or {})

        _log("engine.start", {"phase": state.phase.value, "goal": goal})

        while state.phase != Phase.TERMINAL:
            if state.phase == Phase.DISCOVERY:
                if state.mts.check_sufficiency(cfg.min_mts_confidence):
                    _log(
                        "engine.discovery.sufficient",
                        {
                            "round": state.round,
                            "confidence": state.mts.confidence,
                            "truths": len(state.mts.truths),
                        },
                    )
                    state.phase = Phase.PLANNING
                    state.round = 1
                    continue

                if state.round > cfg.max_discovery_rounds:
                    _log(
                        "engine.discovery.limit_reached",
                        {"max_rounds": cfg.max_discovery_rounds},
                    )
                    state.phase = Phase.TERMINAL
                    state.status = Status.FAILURE
                    return FinalReport(
                        status=state.status,
                        state=state,
                        message="Insufficient MTS after max discovery rounds",
                    )

                request = DiscoveryRequest(
                    goal=state.goal,
                    missing=state.mts.missing,
                    round=state.round,
                    context=state.mts,
                )
                result = self.discoverer.discover(request)
                state.mts.truths |= result.new_truths
                _log(
                    "engine.discovery.round_complete",
                    {
                        "round": state.round,
                        "added_truths": len(result.new_truths),
                        "truths_total": len(state.mts.truths),
                    },
                )
                state.round += 1
                continue

            if state.phase == Phase.PLANNING:
                if not state.mts.sufficient:
                    _log("engine.planning.insufficient_mts")
                    state.phase = Phase.TERMINAL
                    state.status = Status.FAILURE
                    return FinalReport(
                        status=state.status,
                        state=state,
                        message="Planning entered with insufficient MTS",
                    )

                planning_request = PlanningRequest(
                    goal=state.goal, mts=state.mts, previous_dag=state.dag
                )
                planning_result = self.planner.plan(planning_request)

                _log(
                    "engine.planning.result",
                    {
                        "estimated_rounds": planning_result.estimated_rounds,
                        "nodes": len(planning_result.dag.nodes),
                    },
                )

                if planning_result.estimated_rounds > cfg.max_execution_rounds:
                    _log(
                        "engine.planning.too_many_rounds",
                        {
                            "estimated": planning_result.estimated_rounds,
                            "max": cfg.max_execution_rounds,
                        },
                    )
                    state.phase = Phase.TERMINAL
                    state.status = Status.FAILURE
                    return FinalReport(
                        status=state.status,
                        state=state,
                        message="Planner estimated rounds exceed execution limit",
                    )

                if not planning_result.dag.is_acyclic():
                    _log("engine.planning.cycle_detected")
                    state.phase = Phase.TERMINAL
                    state.status = Status.FAILURE
                    return FinalReport(
                        status=state.status,
                        state=state,
                        message="Planner produced cyclic DAG",
                    )

                state.dag = planning_result.dag
                state.phase = Phase.EXECUTION
                state.round = 1
                _log("engine.execution.start", {"tasks": len(state.dag.nodes)})
                continue

            if state.phase == Phase.EXECUTION:
                if state.round > cfg.max_execution_rounds:
                    _log(
                        "engine.execution.timeout",
                        {"max_rounds": cfg.max_execution_rounds},
                    )
                    state.phase = Phase.TERMINAL
                    state.status = Status.TIMEOUT
                    return FinalReport(
                        status=state.status,
                        state=state,
                        message="Execution timeout (max rounds exceeded)",
                    )

                assert state.dag is not None

                ready_ids = state.dag.get_ready_tasks(state.completed_nodes)
                if not ready_ids:
                    state.phase = Phase.EVALUATION
                    _log("engine.execution.no_ready_tasks", {"round": state.round})
                    continue

                selected_ids = list(ready_ids)[: cfg.max_tasks_per_round]
                for node_id in selected_ids:
                    node = state.dag.nodes[node_id]
                    assert node.type == "atomic"
                    job = WorkerJob(node=node, context={}, timeout=60.0)
                    worker_result = self.worker.execute(job)
                    assert worker_result.node_id in state.dag.nodes
                    state.results[node_id] = worker_result
                    if worker_result.status == "success":
                        state.completed_nodes.add(node_id)

                    _log(
                        "engine.execution.task_complete",
                        {
                            "round": state.round,
                            "node_id": node_id,
                            "status": worker_result.status,
                        },
                    )

                state.phase = Phase.EVALUATION
                continue

            if state.phase == Phase.EVALUATION:
                report = self.evaluator.evaluate(state)
                _log(
                    "engine.evaluation.report",
                    {
                        "overall": report.overall,
                        "completed": len(state.completed_nodes),
                        "truths": len(state.mts.truths),
                    },
                )

                if report.overall == "early_success":
                    state.phase = Phase.TERMINAL
                    state.status = Status.SUCCESS
                    _log("engine.terminal.early_success")
                    return FinalReport(status=state.status, state=state, message="Early success")

                if report.overall == "unrecoverable_fail":
                    state.phase = Phase.TERMINAL
                    state.status = Status.FAILURE
                    _log("engine.terminal.unrecoverable_fail")
                    return FinalReport(
                        status=state.status,
                        state=state,
                        message="Unrecoverable failure",
                    )

                if report.overall == "need_fix":
                    state.phase = Phase.TERMINAL
                    state.status = Status.FAILURE
                    _log("engine.terminal.need_fix")
                    return FinalReport(
                        status=state.status,
                        state=state,
                        message="NeedFix not supported in MVP",
                    )

                if report.overall == "ok":
                    if state.dag and len(state.completed_nodes) < len(state.dag.nodes):
                        state.round += 1
                        state.phase = Phase.EXECUTION
                        _log("engine.execution.next_round", {"round": state.round})
                        continue
                    state.phase = Phase.TERMINAL
                    state.status = Status.SUCCESS
                    _log("engine.terminal.success")
                    return FinalReport(
                        status=state.status, state=state, message="All tasks completed with OK"
                    )

        _log("engine.terminal.exit_loop", {"status": state.status.value})
        return FinalReport(status=state.status, state=state, message="Exited loop")
