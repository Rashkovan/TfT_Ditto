---
name: Project Ditto overview
description: Ditto is a Pyro DAG visualizer; key facts the conductor needs at hand
type: project
---

Ditto is a Dash + Cytoscape DAG visualizer for Pyro probabilistic models.

**Why:** This is a class project for "Tools For Thought"; the user iterates feature-by-feature against a phased roadmap in `ProjectSpecs/roadmap.md`.

**How to apply:**
- The console script entry point is `ditto-viz` (defined in `pyproject.toml`'s `[project.scripts]`), NOT `ditto`. Don't tell the user to run `ditto file.py`.
- Three valid annotation tags: `prior`, `latent`, `observed`. The old `approx` tag is gone (auto-guide replaces it).
- Required user-file callables: `model(...)` and `get_data() -> (args, kwargs)`. Missing `get_data` raises a friendly AttributeError.
- Tests live under `tests/` with fixtures under `tests/fixtures/`. Run with `pytest tests/`.
- The Python venv lives at `pyenv/`; activate with `source pyenv/bin/activate` before running anything.
