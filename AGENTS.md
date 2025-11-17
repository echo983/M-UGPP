# Repository Guidelines

## Project Structure & Module Organization
- Repo is currently a clean slate; organize Python packages under `src/` (`src/<package>/__init__.py`). Use `pyproject.toml` at root for metadata and tooling config.
- Keep tests in `tests/`, mirroring package paths (e.g., `tests/<package>/test_feature.py`); place shared fixtures in `tests/fixtures`.
- Store automation utilities in `scripts/` (make them executable and self-documenting); keep documentation assets in `docs/` or `docs/assets`.
- Configuration files (`.env.example`, `.editorconfig`, `pyproject.toml`, CI configs) live at the root; sample data goes to `data/` and must exclude secrets.

## Build, Test, and Development Commands
- Create and activate a virtualenv (`python -m venv .venv && source .venv/bin/activate`); install dependencies via `pip install -r requirements.txt` or `pip install -e .[dev]`.
- Use `pytest` for the test suite; run `pytest` for unit tests and `pytest -m "not slow"` to skip slow marks. Prefer a single entrypoint like `make test`.
- Run `ruff check` for lint, `ruff format` or `black` for formatting, and `isort` if defined; add a `make lint` target that chains them.
- Place local setup helpers in `scripts/` and ensure they print concise `--help` usage.

## Coding Style & Naming Conventions
- Indent with 4 spaces for Python and 2 spaces for YAML; trim trailing whitespace and keep POSIX newlines.
- Use descriptive, typed names: functions as verbs, classes in PascalCase, modules and files in snake_case; avoid abbreviations except common acronyms.
- Honor automated tools configured in `pyproject.toml` (`ruff`, `black`, `isort`) plus `.editorconfig`; run them before commits.

## Testing Guidelines
- Co-locate tests mirroring module names (`foo.py` → `tests/test_foo.py`, `utils.ts` → `tests/utils.test.ts`).
- Aim for high coverage of core logic; add regression tests for every bug fix.
- Mark slow or integration tests and keep fixtures deterministic; ensure the full suite runs with a single command such as `make test`.

## Commit & Pull Request Guidelines
- Commit messages: short, imperative subject (<72 chars) with optional body explaining why, risks, and follow-ups; keep diffs scoped.
- PRs should describe intent, local verification steps, linked issues/tickets, and UI evidence (screenshots or recordings) when relevant.
- Update docs and examples alongside code; request review only after tests and linters are green.

## Security & Configuration Tips
- Never commit secrets; provide `.env.example` and rely on environment variables for runtime config.
- Restrict scripts that handle credentials and scrub logs of sensitive data; rotate keys used in testing.
