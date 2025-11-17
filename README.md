# Minimal UGPP Demo

This repository contains a runnable, minimal implementation of the UGPP (Understanding, Guidance, Planning, Production) engine with simple role implementations and a CLI entrypoint.

## Quick start

1. Create a virtual environment and install the package in editable mode:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

2. Run the pipeline with the `ugpp` command:

```bash
ugpp "Ship a demo feature"
```

Pass `--dump-json` to capture machine-readable output, or add repeated `--truth` and `--need` flags to seed the MTS for automated tests and debugging.

Example with seeded needs and truths:

```bash
ugpp "Improve docs" \
  --need need-docs|Clarify contribution steps|0.8|medium|docs-gap \
  --truth t1|Docs outline drafted|0.75
```

Environment variables `OPENAI_API_KEY`, `ADA_BRAIN_MODEL`, and `ADA_CODER_MODEL` are not required for the demo roles but can be exported for downstream integrations:

```bash
export OPENAI_API_KEY=sk-...your-key...
export ADA_BRAIN_MODEL=gpt-5.1
export ADA_CODER_MODEL=gpt-5-mini
```
