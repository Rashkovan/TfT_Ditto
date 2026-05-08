---
name: Phase 6 architectural decisions
description: How Ditto's interactive GUI features are structured (dcc.Store + server-side runtime state)
type: project
---

Phase 6 of Ditto adds runtime model loading, drag-drop, prior editing, data upload, and refresh.

**Why:** Users wanted to explore multiple models in one session and tweak priors / data without restarting the server. Roadmap explicitly required `dcc.Store`-based state (no `dcc.Interval` polling).

**How to apply:**
- `cli.py`: `filepath` is `nargs="?"`; when None, `create_dash_app(None, None, config)` launches an empty canvas. `--export` and `--dry-run` still require a filepath and error out otherwise.
- `visualizer.py` holds all Phase 6 logic — no other module needed changes. The only spot where state spans process boundaries is the `_RuntimeState` instance closed over by the callbacks (holds `DittoGraph`, `user_module`, last `InferenceResult`, current `filepath`); JSON-friendly bits go in `dcc.Store`.
- The Cytoscape component is always rendered; an `empty-placeholder` div sits next to it and `display: none` toggles between them. This avoids "no such id" errors when callbacks output `cytoscape.elements` while the canvas is empty.
- "Apply Prior" only re-samples the prior for the edited variable. Posterior re-inference is gated behind the explicit Refresh button so SVI doesn't run on every keystroke.
- Pending prior edits are preserved across Refresh (they live in `prior-edit-store`), but reset on a brand-new file load.
- `latent` tags imply BOTH prior and posterior plots — `show_prior = bool(effective_tags & {"prior", "latent"})`. There's a fallback to `frozenset({variable.tag})` when callers construct `AnnotatedVariable` without explicit `tags` (tests do this).
