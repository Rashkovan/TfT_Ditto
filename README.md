# Ditto

A DAG visualization tool for [Pyro](https://pyro.ai/) probabilistic models.

Ditto reads a Pyro source file annotated with `# !Ditto:` comments, builds the
implicit dependency graph, runs the model (sampling priors and, optionally,
running SVI with an auto-generated guide for any `latent` variables), and serves an interactive Dash +
Cytoscape web app where each node carries a hover-on KDE/histogram of its
samples.

## Quick start

```bash
pip install -e .
ditto-viz path/to/your_model.py
```

then open http://localhost:8050.

### Annotating your model

Place a `# !Ditto: <tag>` comment on the line directly above each assignment
you want to surface in the DAG. Tags are:

- `prior` — a `pyro.distributions` object you want to sample from directly;
  the tooltip shows its prior distribution.
- `latent` — a latent variable to be inferred. The annotated expression
  may be either a bare distribution (`dist.Normal(0., 1.)`) or a
  `pyro.sample("name", dist.X(...))` call (useful when the annotation lives
  inside your `model` function). Ditto auto-creates an `AutoNormal(model)`
  guide for bare-distribution latents, or `AutoNormalizingFlow(model, ...)`
  if any latent is declared via `pyro.sample(...)`, and runs SVI. The
  tooltip shows *both* the prior distribution (sampled from this
  expression) and the posterior distribution (from `Predictive` over the
  trained guide), stacked vertically.
- `observed` — a site that consumes other variables (typically your
  likelihood); the tooltip shows the posterior predictive distribution.

Example (see `tests/fixtures/simple_model.py`):

```python
# !Ditto: latent
mu = dist.Normal(0., 1.)

# !Ditto: observed
x = dist.Normal(mu, 1.)
```

Your file must also define a top-level `model(...)` callable and a
`get_data() -> (args, kwargs)` helper. You no longer need to define a
guide — Ditto builds one automatically whenever any `latent` variable is
present.

## Configuration

Defaults live in `ditto.yaml`. CLI flags:

| flag | default | meaning |
|------|---------|---------|
| `--config` | `ditto.yaml` | path to a YAML config |
| `--port` | `8050` | port for the Dash server |
| `--export PATH` | _off_ | write a static PNG of the DAG to `PATH` |
| `--dry-run` | _off_ | run pipeline but don't start the server |

## Programmatic use

```python
import ditto
ditto.run("path/to/your_model.py")  # or use: ditto-viz path/to/your_model.py from the CLI
```

## Tests

```bash
pip install -e ".[dev]"
pytest tests/
```

---

## Conductor Notes

This section is maintained by the Ditto build conductor. It records major
design decisions, obstacles encountered, and follow-up suggestions.

### Major design decisions

- **Side-panel affordances are tag-gated (2026-05-08).** "Edit prior"
  and "Data" panels used to render unconditionally for every selected
  node, which was confusing on `latent` and `observed` variables (no
  prior to edit) and on `prior` variables (no observations to upload).
  The side-panel renderer now gates on the variable's tag set: the
  prior textarea is shown only when `"prior"` is in `tags`, and the
  data uploader is shown only when `"observed"` is in `tags`. Hidden
  stub components (`display: none`) keep the IDs in the layout for
  every render so the per-component callbacks stay happy. **Edits
  persist across Refresh (2026-05-08):** typing into the prior
  textarea now writes to `prior-edit-store` via a dedicated capture
  callback. The Refresh handler flushes any pending edits to the
  source `.py` (via `_rewrite_prior_in_file`) **before** re-running
  the parse/build/inference pipeline, so a typed-but-not-Apply'd edit
  still propagates through the model on Refresh. After the pipeline
  runs, `prior-edit-store` is cleared because the freshly re-parsed
  elements are now the source of truth. To avoid focus loss while
  typing, `prior-edit-store` is now a `State` (not an `Input`) of the
  side-panel renderer; the renderer only re-fires on hover, selection,
  model, or data-store changes.

- **Phase 6 GUI enhancements (2026-05-07).** `cli.py` now accepts an
  optional `filepath`; when omitted, the Dash app launches in an empty
  state and the user loads a file at runtime via drag-and-drop or a
  native "Open File…" button (`tkinter.filedialog`). All runtime mutable
  state — current model, edited prior expressions, uploaded data,
  selected node — lives in `dcc.Store` components. Heavy server-side
  objects (`DittoGraph`, `user_module`, `InferenceResult`) are kept in a
  `_RuntimeState` instance closed over by the callbacks. The "Apply
  Prior" button parses the new expression with `ast.parse(expr,
  mode="eval")` for validation, mutates the in-memory variable's
  `raw_expression`, and re-samples only that node's prior; full
  posterior re-inference is gated behind the "Refresh" button so users
  control when expensive SVI runs. Thumbnails are still pre-computed
  once per node (now per refresh) and stored in the elements list, so
  the hover/click callbacks remain matplotlib-free at request time.
  `dcc.Loading` wraps the DAG container as the running-state indicator
  rather than `dcc.Interval` polling, per the architectural rule.
  Backward compatibility: when a filepath is passed at launch, the app
  behaves exactly as it did before — same elements, same hover
  callback, same `--export` and `--dry-run` flags.

- **Latent nodes always get a prior plot (2026-05-07).** The
  `build_cytoscape_elements` predicate `show_prior = "prior" in
  variable.tags` was bug-compatible with parser-produced variables only
  when the tag set explicitly included `"prior"`; latent-only tags
  (`{"latent"}`) silently produced empty prior thumbnails despite the
  visualizer test expecting a populated one. Fixed by gating on
  `effective_tags & {"prior", "latent"}` and falling back to
  `frozenset({variable.tag})` when callers (notably tests) construct
  `AnnotatedVariable` without an explicit `tags` field.

- **`pyro.sample(...)` annotations for latent vars (2026-05-02).** Users
  often write `# !Ditto: latent` directly above a `pyro.sample(...)` call
  inside their `model` function rather than above a bare distribution
  expression. Naively `eval`-ing the RHS would execute the sample site and
  return a tensor, breaking `sample_prior`. The engine now detects this via
  `is_pyro_sample_call` (AST check for a top-level `Call` with `func.id ==
  "sample"` or `func.attr == "sample"`) and pulls the second positional
  argument out as the distribution sub-expression to eval instead. The
  parser stamps `AnnotatedVariable.is_sample_call` at parse time so
  downstream logic doesn't re-parse the string. Guide selection follows
  the same predicate: any sample-call latent triggers `AutoNormalizingFlow`
  (with `block_autoregressive` as `init_transform_fn`); otherwise we fall
  back to `AutoNormal` for the cheaper, simpler joint.

- **Tag set refactor (2026-05-02): `prior` / `latent` / `observed`.** The
  old `approx` tag (which required users to write `guide = AutoNormal(model)`
  themselves) was removed. Latent variables are now declared with
  `# !Ditto: latent` on their prior distribution; Ditto auto-creates
  `AutoNormal(user_module.model)` whenever any latent var is present. Latent
  nodes display both prior and posterior thumbnails stacked vertically in
  the hover tooltip. `InferenceResult` now carries `prior_samples` and
  `posterior_samples` as separate dicts (with a `.samples` merged property
  retained for backward compatibility).

- **Two-pass parser (regex + AST).** Python's `ast` module strips comment
  nodes entirely, so the regex pass is *necessary* (not a shortcut) to
  recover the line number of each `# !Ditto:` annotation. The AST pass then
  picks the nearest assignment after each annotation by line number; this
  tolerates blank lines between the comment and its target assignment.
- **Single-target assignments only.** Each annotation must point at an
  `Assign`/`AnnAssign` with exactly one bare-name target. This guarantees a
  one-to-one mapping between annotations and DAG nodes. Tuple unpacking and
  attribute assignments raise a clear `ValueError`.
- **Edge inference by name intersection.** `graph_builder` extracts every
  bare `ast.Name` from the RHS expression and intersects with the set of
  *known annotated names*, then subtracts `{self}`. Library names like
  `torch`, `dist`, `pyro` drop out automatically because they aren't
  annotated; this keeps the rule simple and avoids hand-maintained
  blocklists.
- **Self-references silently ignored** instead of treated as a cycle. A
  variable that mentions its own name in its RHS produces no edge and no
  error. Rationale: writing `a = a + 1` in Pyro source is unusual but not
  semantically a cycle in the *modeling* sense, and erroring would surprise
  users more than helping them.
- **Priors take precedence over posterior samples.** When SVI runs, any site
  it produces samples for that already has a prior tensor in `all_samples`
  is discarded for that site. This keeps the displayed prior plots truly
  prior (rather than silently replacing them with posteriors).
- **`obs` keyword stripped before `Predictive`.** The spec calls this out
  explicitly: passing observed data to `Predictive` makes it return the
  observation tensor verbatim. The engine strips `obs` from the kwargs
  copy used for posterior prediction so fresh draws are produced.
- **`poutine.uncondition` for model-internal observations (2026-05-02).**
  When the user conditions inside their `model` (e.g.
  `pyro.sample("obs", ..., obs=data)` rather than passing `obs=` via
  kwargs), `_strip_observed` has nothing to strip and `Predictive` keeps
  returning the fixed data tensor. `collect_posterior_samples` now wraps
  the model in `pyro.poutine.uncondition` before passing it to
  `Predictive`, which strips `obs=` from every internal sample call. The
  legacy `_strip_observed` helper is preserved for backward compatibility
  but no longer called by `run_inference`. The actual observed tensor is
  recovered separately via `extract_observed_data` (a one-shot
  `poutine.trace` over the model) and concatenated onto the posterior
  predictive draws so the visualizer can show both together.
- **Bare `pyro.sample(...)` annotations (2026-05-02).** The parser now
  also matches `ast.Expr` statements wrapping a `(pyro.)sample(...)` call,
  not just `Assign`/`AnnAssign`. The site name is taken from the
  string-literal first argument; non-literal first args raise
  `ValueError` because Ditto needs a stable site key. This is what makes
  `# !Ditto: observed` work above bare conditioning sites inside a
  `model` function.
- **Thumbnails pre-computed at startup.** `build_cytoscape_elements` runs
  matplotlib once per node and ships the base64 PNG inline in the
  Cytoscape node data. The hover callback is then a pure dict lookup —
  matplotlib never runs at request time, which keeps the UI snappy and
  avoids matplotlib's thread-safety footguns under Dash.
- **`cyto.load_extra_layouts()` at module import time.** Without it the
  `dagre` layout silently falls back to `circle`. Putting it at the top of
  `visualizer.py` (immediately after the import) means *any* path through
  the package gets it, including programmatic use via `ditto.run`.
- **`app.run` instead of `app.run_server`.** Dash >= 2.11 deprecated
  `run_server` in favor of `run`. Functionally identical, but `run` is the
  forward-compatible name.

### Obstacles encountered

- The fixture skeleton in the spec has `guide = AutoNormalizingFlow(model)`
  *above* the `def model(...)` block. That parses, but executing it raises
  `NameError` — and the inference tests need to import the module. The
  fixtures here keep the same annotation tags and order but reorder the
  function definitions above the annotated assignments so that the file is
  both parseable *and* importable. Same change applied to
  `linear_regression.py`. Switched the guide class to `AutoNormal` to avoid
  `AutoNormalizingFlow`'s extra setup requirements during testing.
- `pyro.optim.Adam` vs. `torch.optim.Adam` is a real footgun: SVI's
  optimizer interface only matches the Pyro wrapper, and using the torch
  optimizer fails at the first SVI step with an opaque traceback. Called
  out in the engine's docstring.
- **`pyro.optim` is a submodule, not auto-loaded by `import pyro`.**
  Discovered during the 2026-05-02 critic/dev cycle: `pyro.optim.Adam(...)`
  raised `AttributeError: module 'pyro' has no attribute 'optim'`. Fix is
  an explicit `import pyro.optim` at the top of `inference_engine.py`.
- **`argparse --port default=8050` masked YAML config.** The CLI's
  `port = args.port or config["server"]["port"]` always picked the argparse
  default because `8050` is truthy. Switched argparse default to `None` and
  used an explicit `is not None` check so YAML port settings are honored.
- **`eval` mutates the namespace dict** by inserting `__builtins__`. The
  module-level `EVAL_NAMESPACE` is now copied per call so state doesn't
  leak across `sample_prior` invocations.
- **Defensive empty-sample plotting.** `plot_variable` now short-circuits
  on a 0-element array with a "no samples" placeholder rather than passing
  through to KDE/histogram code that would crash on empty input.
- **CLI traceback printing.** The top-level `except Exception` previously
  swallowed the traceback, leaving only a one-line message. We now also
  call `traceback.print_exc()` so the underlying error is debuggable
  without re-running with `-X dev`.

### Suggestions for follow-up features

- **Posterior-vs-prior overlay.** When a site has both prior draws and
  posterior draws available, plot them on the same axes (different alpha)
  so users can see how SVI moved the distribution.
- **Click-to-pin tooltips.** Currently the right-pane plot follows mouseover
  and resets when the cursor leaves. A click-to-pin mode would let users
  compare two nodes side by side.
- **Live-reload on source change.** Watch the user's source file with
  `watchdog`; on change, re-parse, re-build the graph, and push an updated
  Cytoscape `elements` payload via Dash's clientside callbacks.
- **MCMC backend.** Currently inference is SVI-only. A `# !Ditto: mcmc`
  tag (or a config flag) could route to NUTS/HMC for cases where the user
  wants asymptotically exact samples.
- **Node summary stats.** Show mean/std/quantiles in the tooltip alongside
  the KDE plot.
- **Cycle visualization.** When the parser detects a cycle, render the
  offending edges in red rather than just raising — easier to debug for
  users with large models.
