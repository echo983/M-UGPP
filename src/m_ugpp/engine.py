"""UGPP engine coordinating discovery, planning, execution, and evaluation."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

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

    def run(self, goal: str, initial_mts: Optional[MTS] = None) -> FinalReport:
        mts = initial_mts or MTS()
        state = UGPPState(phase=Phase.DISCOVERY, round=1, goal=goal, mts=mts)
        cfg = self.config

        while state.phase != Phase.TERMINAL:
            if state.phase == Phase.DISCOVERY:
                if state.mts.check_sufficiency(cfg.min_mts_confidence):
                    state.phase = Phase.PLANNING
                    state.round = 1
                    continue

                if state.round > cfg.max_discovery_rounds:
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
                state.round += 1
                continue

            if state.phase == Phase.PLANNING:
                if not state.mts.sufficient:
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

                if planning_result.estimated_rounds > cfg.max_execution_rounds:
                    state.phase = Phase.TERMINAL
                    state.status = Status.FAILURE
                    return FinalReport(
                        status=state.status,
                        state=state,
                        message="Planner estimated rounds exceed execution limit",
                    )

                if not planning_result.dag.is_acyclic():
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
                continue

            if state.phase == Phase.EXECUTION:
                if state.round > cfg.max_execution_rounds:
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

                state.phase = Phase.EVALUATION
                continue

            if state.phase == Phase.EVALUATION:
                report = self.evaluator.evaluate(state)

                if report.overall == "early_success":
                    state.phase = Phase.TERMINAL
                    state.status = Status.SUCCESS
                    return FinalReport(status=state.status, state=state, message="Early success")

                if report.overall == "unrecoverable_fail":
                    state.phase = Phase.TERMINAL
                    state.status = Status.FAILURE
                    return FinalReport(
                        status=state.status,
                        state=state,
                        message="Unrecoverable failure",
                    )

                if report.overall == "need_fix":
                    state.phase = Phase.TERMINAL
                    state.status = Status.FAILURE
                    return FinalReport(
                        status=state.status,
                        state=state,
                        message="NeedFix not supported in MVP",
                    )

                if report.overall == "ok":
                    if state.dag and len(state.completed_nodes) < len(state.dag.nodes):
                        state.round += 1
                        state.phase = Phase.EXECUTION
                        continue
                    state.phase = Phase.TERMINAL
                    state.status = Status.SUCCESS
                    return FinalReport(
                        status=state.status, state=state, message="All tasks completed with OK"
                    )

        return FinalReport(status=state.status, state=state, message="Exited loop")
