"""Concrete role implementations for a runnable UGPP demo pipeline."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import subprocess
import time
from typing import Any, Dict, Iterable, Optional, Sequence, Set, Tuple

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
from .gpt import GPTClient, GPTError
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
            report="Simple discoverer placeholder",
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


def _make_truth(item: Dict[str, Any]) -> Truth:
    """Build a Truth object from a GPT-provided dictionary."""

    return Truth(
        id=item.get("id", f"truth-{int(time.time())}"),
        statement=item.get("statement", ""),
        sources=frozenset(item.get("sources") or _default_sources()),
        confidence=float(item.get("confidence", 0.7)),
        timestamp=datetime.utcnow(),
        verifiable=bool(item.get("verifiable", True)),
    )


@dataclass
class CommandWorker(Worker):
    """Execute bash or python tasks and capture their output."""

    default_workdir: str = "."
    fallback: SimpleWorker = field(default_factory=SimpleWorker)

    def _run_subprocess(self, command: Sequence[str], cwd: Path, timeout: float) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            command,
            cwd=str(cwd),
            capture_output=True,
            text=True,
            timeout=timeout,
        )

    def execute(self, job: WorkerJob) -> WorkerResult:  # noqa: D401
        """Run script tasks, falling back to the simple worker for generic nodes."""

        node = job.node
        executor = node.metadata.get("executor")
        payload = node.metadata.get("payload")
        workdir = Path(node.metadata.get("workdir", self.default_workdir)).expanduser().resolve()
        start = time.perf_counter()

        if executor not in {"bash", "python"} or not payload:
            return self.fallback.execute(job)

        cmd = ["bash", "-lc", payload] if executor == "bash" else ["python", "-c", payload]

        try:
            proc = self._run_subprocess(cmd, cwd=workdir, timeout=job.timeout)
            duration = time.perf_counter() - start
            status = "success" if proc.returncode == 0 else "failure"
            node.status = status
            artifacts: Dict[str, Any] = {
                "executor": executor,
                "command": payload,
                "workdir": str(workdir),
                "returncode": proc.returncode,
                "stdout": proc.stdout,
                "stderr": proc.stderr,
            }
            error = None if status == "success" else f"Non-zero exit status {proc.returncode}"
            return WorkerResult(
                node_id=node.id,
                status=status,
                artifacts=artifacts,
                execution_time=duration,
                error=error,
                goal_met=status == "success" and proc.returncode == 0,
            )
        except subprocess.TimeoutExpired as exc:
            duration = time.perf_counter() - start
            node.status = "timeout"
            return WorkerResult(
                node_id=node.id,
                status="timeout",
                artifacts={
                    "executor": executor,
                    "command": payload,
                    "workdir": str(workdir),
                    "stdout": exc.stdout or "",
                    "stderr": exc.stderr or "",
                },
                execution_time=duration,
                error=f"Task exceeded timeout of {job.timeout}s",
            )
        except Exception as exc:  # pragma: no cover - defensive catch for unexpected issues
            duration = time.perf_counter() - start
            node.status = "failure"
            return WorkerResult(
                node_id=node.id,
                status="failure",
                artifacts={
                    "executor": executor,
                    "command": payload,
                    "workdir": str(workdir),
                    "stdout": "",
                    "stderr": str(exc),
                },
                execution_time=duration,
                error=str(exc),
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


# GPT-powered roles --------------------------------------------------------


@dataclass
class GPTDiscoverer(Discoverer):
    """Use GPT to propose truths and confidence for the goal with active probes."""

    model: str = "gpt-5-nano"
    confidence_floor: float = 0.65
    probe_commands: Optional[Tuple[Tuple[str, str], ...]] = None
    workdir: str = "."
    max_workers: int = 32
    enable_probes: bool = True

    def discover(self, request: DiscoveryRequest) -> DiscoveryResult:  # noqa: D401
        cache_key = f"discover|{self.model}|{request.goal[:64]}"
        client = GPTClient(model=self.model)
        missing_questions = [need.question for need in request.missing] or ["What is the key information?"]
        probe_results: Dict[str, Dict[str, Any]] = {}

        if self.enable_probes and self.probe_commands:
            executor = CommandWorker(default_workdir=self.workdir)
            jobs = list(self.probe_commands)
            with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
                future_map = {
                    pool.submit(executor.execute, WorkerJob(node=TaskNode(id=f"probe-{i}", type="atomic", preconditions=[], postconditions=[], metadata={"executor": ex, "payload": cmd, "workdir": self.workdir, "goal": request.goal}), context={}, timeout=30.0)): (ex, cmd)
                    for i, (ex, cmd) in enumerate(jobs, start=1)
                }
                for future in as_completed(future_map):
                    ex, cmd = future_map[future]
                    try:
                        res = future.result()
                        probe_results[cmd] = res.artifacts
                    except Exception as exc:  # pragma: no cover - defensive
                        probe_results[cmd] = {"error": str(exc)}

        payload = client.chat_json(
            [
                {
                    "role": "system",
                    "content": "You help gather facts to satisfy missing knowledge before planning. "
                    "Use probe outputs if provided. Return concise, verifiable statements as JSON.",
                },
                {
                    "role": "user",
                    "content": (
                        f"Goal: {request.goal}\n"
                        f"Round: {request.round}\n"
                        f"Missing questions: {missing_questions}\n"
                        f"Probe outputs: {probe_results}\n"
                        "Respond JSON: {\"truths\": [{\"id\": \"t1\", \"statement\": \"...\", \"confidence\": 0.8}], \"report\": \"short summary\"}"
                    ),
                },
            ],
            prompt_cache_key=cache_key,
        )

        truths_raw = payload.get("truths") or []
        truths = {_make_truth(item) for item in truths_raw}
        confidence = min(max(item.get("confidence", 0.7) for item in truths_raw), 1.0) if truths_raw else 0.7
        confidence = max(confidence, self.confidence_floor)
        report = payload.get("report")
        artifacts: Dict[str, Any] = {"probes": probe_results}
        return DiscoveryResult(
            new_truths=truths, confidence=confidence, sources_used=_default_sources(), report=report, artifacts=artifacts
        )


@dataclass
class GPTPlanner(Planner):
    """Use GPT to create an executable task DAG (bash/python steps)."""

    model: str = "gpt-5.1"
    workdir: str = "."
    max_steps: int = 8
    preset_tasks: Optional[Sequence[Tuple[str, str]]] = None

    def plan(self, request: PlanningRequest) -> PlanningResult:  # noqa: D401
        if self.preset_tasks:
            tasks = [{"executor": ex, "payload": pl} for ex, pl in self.preset_tasks]
        else:
            cache_key = f"planner|{self.model}|{request.goal[:64]}"
            client = GPTClient(model=self.model)
            prompt = (
                "You are a system operator creating a short DAG to accomplish a goal. "
                "Each node must include executor (bash|python), payload (cmd), expected_outputs (dict), resources (list), failure_conditions (list), parallelizable (bool). "
                "Return JSON: {\"tasks\": [{\"executor\": \"bash|python\", \"payload\": \"cmd\", \"expected_outputs\": {}, \"resources\": [], \"failure_conditions\": [], \"parallelizable\": true}]}. "
                "Keep tasks <= %d, prefer side-effect-free operations unless necessary."
                % self.max_steps
            )
            payload = client.chat_json(
                [
                    {"role": "system", "content": prompt},
                    {
                        "role": "user",
                        "content": f"Goal: {request.goal}\nKnown truths: {[t.statement for t in request.mts.truths]}",
                    },
                ],
                temperature=0.1,
                prompt_cache_key=cache_key,
            )
            tasks = payload.get("tasks") or []
        nodes: Dict[str, TaskNode] = {}
        edges: Set[tuple[str, str]] = set()
        resolved_workdir = str(Path(self.workdir).expanduser().resolve())

        for index, task in enumerate(tasks[: self.max_steps], start=1):
            executor = task.get("executor", "bash").lower()
            payload_text = task.get("payload", "").strip()
            if executor not in {"bash", "python"} or not payload_text:
                continue
            node_id = f"{executor}-task-{index}"
            nodes[node_id] = TaskNode(
                id=node_id,
                type="atomic",
                preconditions=[],
                postconditions=[],
                metadata={
                    "executor": executor,
                    "payload": payload_text,
                    "workdir": resolved_workdir,
                    "goal": request.goal,
                },
                expected_outputs=task.get("expected_outputs") or {},
                resources=task.get("resources") or [],
                failure_conditions=task.get("failure_conditions") or [],
                parallelizable=bool(task.get("parallelizable", True)),
            )
            if index > 1:
                prev = f"{list(nodes.keys())[-2]}"
                edges.add((prev, node_id))

        if not nodes:
            # Fallback to a single no-op marker to avoid empty DAGs
            fallback = TaskNode(
                id="task-noop",
                type="atomic",
                preconditions=[],
                postconditions=[],
                metadata={"executor": "bash", "payload": "echo 'No tasks generated'", "workdir": resolved_workdir},
            )
            nodes[fallback.id] = fallback

        estimated_rounds = max(1, len(nodes))
        return PlanningResult(
            dag=DAG(nodes=nodes, edges=edges),
            estimated_rounds=estimated_rounds,
            confidence=0.8,
        )


@dataclass
class GPTWorker(Worker):
    """Use GPT to fill missing executors/payloads, then execute commands."""

    model: str = "gpt-5-mini"
    command_worker: CommandWorker = field(default_factory=CommandWorker)

    def _complete_job(self, job: WorkerJob) -> tuple[Optional[str], Optional[str]]:
        """Ask GPT to propose executor/payload if missing."""

        node = job.node
        executor = node.metadata.get("executor")
        payload = node.metadata.get("payload")
        if executor and payload:
            return executor, payload

        cache_key = f"worker|{self.model}|{node.metadata.get('goal','')[:64]}"
        client = GPTClient(model=self.model)
        result = client.chat_json(
            [
                {"role": "system", "content": "Pick a short bash command or python snippet to advance the goal."},
                {
                    "role": "user",
                    "content": f"Goal: {node.metadata.get('goal')} | Node id: {node.id} | Current metadata: {node.metadata}",
                },
            ],
            prompt_cache_key=cache_key,
        )
        executor = (result.get("executor") or "bash").lower()
        payload = result.get("payload")
        return executor, payload

    def execute(self, job: WorkerJob) -> WorkerResult:  # noqa: D401
        node = job.node
        try:
            executor, payload = self._complete_job(job)
            node.metadata["executor"] = executor
            node.metadata["payload"] = payload
        except GPTError as exc:
            node.status = "failure"
            return WorkerResult(
                node_id=node.id,
                status="failure",
                artifacts={"stdout": "", "stderr": str(exc), "executor": None, "command": None},
                execution_time=0.0,
                error=str(exc),
            )

        return self.command_worker.execute(job)


@dataclass
class GPTEvaluator(Evaluator):
    """Use GPT to summarize progress and decide whether to continue."""

    model: str = "gpt-5-mini"

    def evaluate(self, state: "UGPPState") -> EvalReport:  # noqa: D401
        mts = state.mts
        dag = state.dag
        all_completed = bool(dag) and len(state.completed_nodes) == len(dag.nodes)
        remaining = bool(dag) and len(state.completed_nodes) < len(dag.nodes)
        summary = {
            "goal": state.goal,
            "completed": list(state.completed_nodes),
            "truths": [truth.statement for truth in mts.truths],
            "missing": [need.question for need in mts.missing],
            "dag_nodes": list(dag.nodes.keys()) if dag else [],
        }

        try:
            cache_key = f"evaluator|{self.model}|{state.goal[:64]}"
            client = GPTClient(model=self.model)
            payload = client.chat_json(
                [
                    {
                        "role": "system",
                        "content": "Decide run status. Respond JSON: "
                        '{"overall": "ok|need_fix|unrecoverable_fail|early_success", '
                        '"recommendations": ["..."], "notes": "short note"}',
                    },
                    {"role": "user", "content": f"State summary: {summary}"},
                ],
                prompt_cache_key=cache_key,
            )
            overall = payload.get("overall", "ok")
            recommendations = payload.get("recommendations") or []
            details = {"notes": payload.get("notes", ""), "summary": summary}
        except GPTError as exc:
            overall = "ok" if state.completed_nodes else "need_fix"
            recommendations = ["GPT evaluation failed; fallback heuristic used."]
            details = {"error": str(exc), "summary": summary}

        # Keep progress moving when tasks remain; only halt on explicit failures.
        if overall == "need_fix" and remaining:
            recommendations.append("Continue remaining tasks before final judgment.")
            overall = "ok"

        if all_completed and mts.check_sufficiency():
            overall = "ok"

        return EvalReport(overall=overall, details=details, recommendations=recommendations)


@dataclass
class GPTReporter:
    """LLM-powered reporter to summarize probe and execution artifacts."""

    model: str = "gpt-5-mini"

    def summarize(self, artifacts: Dict[str, Any], goal: str) -> str:
        cache_key = f"reporter|{self.model}|{goal[:64]}"
        client = GPTClient(model=self.model)
        payload = client.chat_json(
            [
                {
                    "role": "system",
                    "content": "Summarize system probe outputs succinctly. Highlight CPU, memory, disks. "
                    "Avoid speculation and keep it under 150 words.",
                },
                {"role": "user", "content": f"Goal: {goal}\nArtifacts: {artifacts}"},
            ],
            prompt_cache_key=cache_key,
        )
        return payload.get("summary") or payload.get("report") or json.dumps(payload)


@dataclass
class CommandPlanner(Planner):
    """Planner that builds executable tasks from provided scripts."""

    tasks: Tuple[Tuple[str, str], ...]
    workdir: str = "."
    fallback: SimplePlanner = field(default_factory=SimplePlanner)

    def plan(self, request: PlanningRequest) -> PlanningResult:  # noqa: D401
        """Return a DAG executing provided tasks or fall back to a single task."""

        if not self.tasks:
            return self.fallback.plan(request)

        nodes: Dict[str, TaskNode] = {}
        edges: Set[tuple[str, str]] = set()

        resolved_workdir = str(Path(self.workdir).expanduser().resolve())
        for index, (executor, payload) in enumerate(self.tasks, start=1):
            node_id = f"{executor}-task-{index}"
            node = TaskNode(
                id=node_id,
                type="atomic",
                preconditions=[],
                postconditions=[],
                metadata={
                    "executor": executor,
                    "payload": payload,
                    "workdir": resolved_workdir,
                    "goal": request.goal,
                },
            )
            nodes[node_id] = node
            if index > 1:
                prev = f"{self.tasks[index - 2][0]}-task-{index - 1}"
                edges.add((prev, node_id))

        estimated_rounds = max(1, len(nodes))
        return PlanningResult(
            dag=DAG(nodes=nodes, edges=edges),
            estimated_rounds=estimated_rounds,
            confidence=0.9,
        )


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


def build_default_roles(
    tasks: Sequence[Tuple[str, str]] | None = None,
    workdir: str = ".",
    use_gpt: bool = True,
    models: Optional[Dict[str, str]] = None,
) -> tuple[Discoverer, Planner, Worker, Evaluator]:
    """Helper to instantiate the default role implementations."""

    command_tasks: tuple[tuple[str, str], ...] = tuple(tasks or ())

    # Default to GPT roles; non-GPT path retained only for debugging.
    model_defaults = {
        "discoverer": "gpt-5-nano",
        "planner": "gpt-5.1",
        "worker": "gpt-5-mini",
        "evaluator": "gpt-5-mini",
    }
    if models:
        model_defaults.update(models)

    if use_gpt:
        planner: Planner = GPTPlanner(
            model=model_defaults["planner"],
            workdir=workdir,
            preset_tasks=command_tasks if command_tasks else None,
        )
        worker: Worker = GPTWorker(
            model=model_defaults["worker"],
            command_worker=CommandWorker(default_workdir=workdir),
        )
        return (
            GPTDiscoverer(model=model_defaults["discoverer"], workdir=workdir),
            planner,
            worker,
            GPTEvaluator(model=model_defaults["evaluator"]),
        )

    planner = CommandPlanner(tasks=command_tasks, workdir=workdir)
    worker = CommandWorker(default_workdir=workdir)
    return (SimpleDiscoverer(), planner, worker, SimpleEvaluator())


def update_mts_from_truths(mts: MTS, new_truths: Iterable[Truth]) -> MTS:
    """Apply discovered truths and recompute sufficiency."""

    mts.truths |= set(new_truths)
    mts.check_sufficiency()
    return mts
