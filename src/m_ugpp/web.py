"""Minimal web interface for interacting with the UGPP demo pipeline."""
from __future__ import annotations

import os
from typing import Any, Dict

from flask import Flask, Response, jsonify, request

from .engine import UGPPEngine
from .roles import build_default_roles, seed_mts
from .types import UGPPConfig


app = Flask(__name__)

STATE: Dict[str, Any] = {
    "api_key": None,
    "logs": [],
}


def _add_log(message: str, details: dict | None = None) -> None:
    STATE.setdefault("logs", []).append({"message": message, "details": details or {}})


@app.get("/")
def index() -> Response:
    """Serve a simple HTML page for running the pipeline."""

    return Response(INDEX_HTML, mimetype="text/html")


@app.post("/api/key")
def set_api_key():
    """Store an API key in memory for demonstration purposes."""

    data = request.get_json(silent=True) or {}
    api_key = str(data.get("api_key", "")).strip() or None
    STATE["api_key"] = api_key
    _add_log("api.key.updated", {"has_key": bool(api_key)})
    return jsonify({"ok": True, "has_key": bool(api_key)})


@app.get("/api/logs")
def get_logs():
    """Return the current in-memory log stream."""

    return jsonify(STATE.get("logs", []))


@app.post("/api/run")
def run_pipeline():
    """Execute the pipeline for a goal and return the report plus logs."""

    data = request.get_json(force=True, silent=True) or {}
    goal = str(data.get("goal", "")).strip()

    if not goal:
        return jsonify({"ok": False, "error": "Goal is required"}), 400

    STATE["logs"] = []
    _add_log("web.run.start", {"goal": goal, "has_api_key": bool(STATE.get("api_key"))})

    config = UGPPConfig(
        max_discovery_rounds=int(data.get("max_discovery_rounds", 3)),
        max_execution_rounds=int(data.get("max_execution_rounds", 10)),
        max_tasks_per_round=int(data.get("max_tasks_per_round", 20)),
        min_mts_confidence=float(data.get("min_mts_confidence", 0.7)),
    )

    discoverer, planner, worker, evaluator = build_default_roles()
    engine = UGPPEngine(
        discoverer=discoverer,
        planner=planner,
        worker=worker,
        evaluator=evaluator,
        config=config,
    )

    final_report = engine.run(goal, initial_mts=seed_mts(goal), log=_add_log)

    _add_log("web.run.complete", {"status": final_report.status.value})
    payload = {
        "ok": True,
        "status": final_report.status.value,
        "message": final_report.message,
        "phase": final_report.state.phase.value,
        "logs": STATE.get("logs", []),
        "completed": list(final_report.state.completed_nodes),
        "mts_truths": len(final_report.state.mts.truths),
        "mts_confidence": final_report.state.mts.confidence,
    }
    return jsonify(payload)


INDEX_HTML = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>UGPP Demo UI</title>
  <style>
    body { font-family: Arial, sans-serif; margin: 2rem; background: #fafafa; color: #222; }
    h1 { margin-bottom: 0.5rem; }
    section { background: #fff; padding: 1rem; border-radius: 8px; box-shadow: 0 2px 8px rgba(0,0,0,0.05); margin-bottom: 1rem; }
    label { display: block; margin: 0.5rem 0 0.25rem; font-weight: 600; }
    input, button, textarea { width: 100%; padding: 0.5rem; font-size: 1rem; }
    button { margin-top: 0.5rem; cursor: pointer; background: #1d70b8; color: #fff; border: none; border-radius: 4px; }
    button:disabled { background: #9ca3af; cursor: not-allowed; }
    #logs { max-height: 320px; overflow-y: auto; background: #0b1725; color: #e5e7eb; padding: 0.5rem; border-radius: 6px; font-family: monospace; }
    .status { margin-top: 0.5rem; padding: 0.5rem; background: #eef2ff; border-radius: 4px; }
  </style>
</head>
<body>
  <h1>Minimal UGPP Web Interface</h1>
  <p>Set an API key, provide a goal, and watch the pipeline run.</p>

  <section>
    <h2>API Key</h2>
    <label for="apiKey">API Key</label>
    <input id="apiKey" type="password" placeholder="Enter API key" />
    <button onclick="saveKey()">Save Key</button>
    <div id="keyStatus" class="status"></div>
  </section>

  <section>
    <h2>Run Pipeline</h2>
    <label for="goal">Goal</label>
    <input id="goal" type="text" placeholder="e.g., Generate a demo report" />
    <button onclick="runGoal()" id="runBtn">Run</button>
    <div id="runStatus" class="status"></div>
  </section>

  <section>
    <h2>Logs</h2>
    <pre id="logs"></pre>
  </section>

  <script>
    async function saveKey() {
      const key = document.getElementById('apiKey').value;
      const res = await fetch('/api/key', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ api_key: key })
      });
      const data = await res.json();
      document.getElementById('keyStatus').innerText = data.has_key ? 'API key saved' : 'API key cleared';
    }

    async function runGoal() {
      const btn = document.getElementById('runBtn');
      btn.disabled = true;
      document.getElementById('runStatus').innerText = 'Running...';
      document.getElementById('logs').innerText = '';

      const goal = document.getElementById('goal').value;
      const res = await fetch('/api/run', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ goal })
      });

      const data = await res.json();
      if (!data.ok) {
        document.getElementById('runStatus').innerText = data.error || 'Failed to run';
        btn.disabled = false;
        return;
      }

      document.getElementById('runStatus').innerText = `Status: ${data.status} — ${data.message}`;
      const lines = data.logs.map(entry => `${entry.message}: ${JSON.stringify(entry.details)}`);
      document.getElementById('logs').innerText = lines.join('\n');
      btn.disabled = false;
    }
  </script>
</body>
</html>
"""


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    app.run(host="0.0.0.0", port=port, debug=False)
