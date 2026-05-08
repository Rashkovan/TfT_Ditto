# Ditto Development Roadmap

Ditto is a DAG visualization tool for Pyro probabilistic models. It parses annotated Python source files, builds dependency graphs, approximates posterior distributions via SVI, and renders KDE/histogram visualizations of each variable in its natural constrained space.

---

## Phase 1: Annotation Parser ✓

### Goals
Parse Python source files to identify variables annotated with `# !Ditto: <tag>` comments, extract variable names, classify tags, and capture the associated expression on the following line.

### Deliverables
- A `parser.py` module that accepts a path to a Python source file and returns a list of `AnnotatedVariable` objects.
- Support for the three core tags: `prior`, `observed`, and `latent`.
- Comma-separated multi-tag annotations: `# !Ditto: prior, latent` produces `tags=frozenset({"prior","latent"})` and a primary display tag derived by priority order (`latent > observed > prior`).
- Handling of two annotatable statement forms: `ast.Assign` / `ast.AnnAssign` (variable name from LHS) and bare `ast.Expr(pyro.sample(...))` expression statements (variable name from the string-literal first argument).
- A structured representation of each annotated variable including: `tag`, `tags`, `name`, `line_number`, `raw_expression`, `is_sample_call`.
- Unit tests covering single-tag and multi-tag annotations, bare `pyro.sample(...)` statements, multiple annotations in one file, and edge cases (blank lines between comment and assignment).

### Implementation Notes
The regex pre-scan **must run first** because Python's `ast` module does not include comment nodes. A single `ast.walk` pass finds the closest annotatable statement after each comment. `is_sample_call` is set to `True` for bare `pyro.sample(...)` expression statements so the inference engine can extract the underlying distribution sub-expression rather than evaluating the whole call.

### Status
**Complete.** `parser.py` implements two-pass parsing with multi-tag and bare-sample-call support.

---

## Phase 2: DAG Construction ✓

### Goals
Build a directed acyclic graph (DAG) representing the dependency relationships among annotated variables, including transitive dependencies through unannotated intermediates.

### Deliverables
- A `graph_builder.py` module accepting a list of `AnnotatedVariable` objects and an optional `filepath`.
- Dependency inference by AST-based name extraction from each variable's RHS expression.
- Transitive resolution through unannotated intermediate assignments: `_collect_intermediates` parses all assignments in the file; `_resolve_deps` walks transitively so a chain like `stress → mu_k (unannotated) → knowledge` produces a direct edge `stress → knowledge`.
- Acyclicity validation via `nx.is_directed_acyclic_graph` / `nx.find_cycle`.
- Unit tests verifying correct edge creation for simple chains, fan-in/fan-out topologies, and transitive intermediate resolution.

### Implementation Notes
Annotated variable names are stripped from the intermediates map before transitive resolution so they always terminate the search. Cycles in unannotated intermediates are broken by a `visited` guard in `_resolve_deps`.

### Status
**Complete.** `graph_builder.py` implements transitive dependency resolution.

---

## Phase 3: Distribution Modeling and Inference ✓

### Goals
For each annotated variable, produce prior and/or posterior samples suitable for visualization. Guides are constructed automatically — no user-declared guide variable is needed.

### Deliverables
- An `inference_engine.py` module with the following pathways:
  - **Prior sampling**: evaluate the distribution expression in a controlled namespace (`pyro`, `dist`, `torch`), draw N samples. For `pyro.sample(...)` expressions, extract the distribution sub-expression first.
  - **Prior predictive pass** (for `pyro.sample(...)` latents): run `Predictive(model, num_samples=N)` to correctly resolve inter-variable dependencies that cannot be eval'd in isolation.
  - **Auto-guide creation**: `AutoNormal` when all latent expressions are bare distributions; `AutoNormalizingFlow` (with `block_autoregressive`) when any latent is a `pyro.sample(...)` call.
  - **Discrete site handling**: `find_discrete_latent_sites` identifies discrete latent sites; `poutine.block` hides them from the autoguide.
  - **SVI training**: `pyro.optim.Adam` + `Trace_ELBO`; `pyro.clear_param_store()` before each run.
  - **Posterior predictive sampling**: `Predictive` wrapping the model in `poutine.uncondition` so all `obs=...` sites are re-sampled.
  - **Observed data surfacing**: actual observed values concatenated onto posterior predictive draws for `observed`-tagged sites.
- `InferenceResult` with `prior_samples`, `posterior_samples`, `losses`, and `num_samples`.

### Status
**Complete.** `inference_engine.py` implements all pathways with auto-guide selection and discrete site handling.

---

## Phase 4: Visualization ✓

### Goals
Build an interactive Dash + Cytoscape web app where distribution plots appear as a side panel on node hover. `latent` nodes show both prior and posterior plots stacked vertically.

### Deliverables
- A `visualizer.py` module with:
  - `plot_variable`: KDE (Gaussian KDE with Silverman bandwidth) or histogram (< 10 unique values, or zero-variance fallback).
  - `build_cytoscape_elements`: converts `DittoGraph` to Cytoscape elements; pre-computes `image_prior` and `image_posterior` base64 thumbnails per node.
  - `_build_stylesheet`: generates the Cytoscape stylesheet (node colors by tag, edge style).
  - `create_dash_app`: flex-row layout with the DAG on the left and a hover side panel on the right; hover callback reads pre-computed image dicts (no matplotlib at callback time).
  - `render_dag_static`: optional static PNG export via `networkx.spring_layout` and `matplotlib`.
- Visual distinction: `prior` nodes blue, `observed` nodes orange, `latent` nodes green.
- Graceful degradation when a variable has no samples.

### Status
**Complete.** `visualizer.py` implements dual prior/posterior plots for `latent` nodes and a flex side-panel layout.

---

## Phase 5: UX, CLI, and Integration ✓

### Goals
Provide a polished command-line interface, configuration file support, and robust defaults.

### Deliverables
- A `cli.py` module with `DEFAULT_CONFIG` (all built-in defaults, so running without a `ditto.yaml` works), `_deep_merge` (layers user YAML over defaults), and a flat `argparse` structure.
- Positional `filepath` argument; flags: `--config`, `--port`, `--export`, `--dry-run`.
- `ditto.yaml` configuration schema for SVI steps, learning rate, sample count, server port, and color palette.
- `app.run(debug=False, port=port)` as the final blocking step (Dash >= 2.11).

### Status
**Complete.** `cli.py` implements all deliverables.

---

## Phase 6: GUI Enhancements (Planned)

### Goals
Remove the hard requirement to supply a model file at launch time and add interactive controls that let users inspect and modify their model without editing source files.

### Features

#### 6.1 Start Without a File
Allow `ditto` to launch without a positional `filepath` argument. When no file is provided, the app should open with an empty canvas showing a prompt ("Drop a .py file here or click to browse"). Inference and DAG rendering are deferred until a file is provided.

**Implementation sketch:**
- Make `filepath` optional in `_build_arg_parser` (`nargs="?"`, `default=None`).
- In `main`, skip the parse/build/inference pipeline when `filepath is None` and launch the app in "empty" state.
- The Dash layout must handle a `None` graph gracefully (empty elements list, placeholder text).

#### 6.2 Drag-and-Drop `.py` File
Allow the user to drag a `.py` file from their file manager onto the Dash app window to load it.

**Implementation sketch:**
- Add a `dcc.Upload` component (or a styled `html.Div` with `dcc.Upload`) as an overlay on the canvas area.
- Wire a callback on `dcc.Upload`'s `contents` input that: decodes the base64 file content, writes it to a temp file, re-runs the full parse/build/inference pipeline, and updates the `cytoscape` `elements` and `tooltip-div`.
- Store the current model state (graph, inference result, config) in a `dcc.Store` component so multiple callbacks can access it without re-running inference.

#### 6.3 System File Browser
Add a button that opens the system's native file chooser dialog, allowing the user to navigate their directory tree to select a `.py` file.

**Implementation sketch:**
- A native file dialog cannot be opened from within a browser-side Dash callback. Use `tkinter.filedialog.askopenfilename` (or `PyQt` / `wx`) in a server-side callback triggered by a button click.
- The chosen path is returned to the client via a hidden `dcc.Store`, which triggers the same load pipeline as the drag-and-drop handler.
- Note: this requires the Dash server and the user's desktop to be on the same machine (standard local-dev usage).

#### 6.4 GUI Editable Priors
Let users change the prior distribution expression for any `prior` or `latent` variable directly in the app, without editing the source file.

**Implementation sketch:**
- Add an `html.Textarea` or `dcc.Input` in the side panel that shows the current `raw_expression` when a node is selected (on click, not just hover).
- An "Apply" button triggers a server-side callback that: updates the in-memory `AnnotatedVariable.raw_expression`, re-samples the prior, updates `InferenceResult.prior_samples` for that variable, and refreshes the node's `image_prior` thumbnail.
- Store the mutable model state in a `dcc.Store` so edits persist across callbacks.

#### 6.5 GUI Loadable Data
Allow users to load a new data file (CSV, NPY, or PKL) to replace the dataset returned by `get_data()` without restarting the server.

**Implementation sketch:**
- Add a `dcc.Upload` component in a "Data" tab or side drawer.
- On upload, the server-side callback parses the file (using `pandas`, `numpy.load`, or `pickle.load` as appropriate), updates the model args, re-runs SVI and posterior sampling, and refreshes all posterior thumbnails.
- The expected data format (shape, column names) should be displayed alongside the upload widget so the user knows what to provide.

#### 6.6 Refresh Button — Propagate Prior/Data Changes
A single "Refresh" button that re-runs the full inference pipeline (prior sampling + SVI + posterior predictive) using the current in-memory state of all priors and data.

**Implementation sketch:**
- Add an `html.Button("Refresh", id="refresh-btn")` in the toolbar.
- Wire a callback on its `n_clicks` input that reads the current priors and data from `dcc.Store`, calls `run_inference`, updates all Cytoscape node thumbnails, and resets the side panel.
- The button should be disabled (greyed out) while inference is running; use a `dcc.Interval` or a background thread with a shared flag to track running state and re-enable the button on completion.

---

## Summary Timeline

| Phase | Description | Status |
|-------|-------------|--------|
| 1 | Annotation Parser | Complete |
| 2 | DAG Construction | Complete |
| 3 | Distribution Modeling and Inference | Complete |
| 4 | Visualization | Complete |
| 5 | UX / CLI / Integration | Complete |
| 6.1 | Start without a file | Planned |
| 6.2 | Drag-and-drop `.py` file | Planned |
| 6.3 | System file browser | Planned |
| 6.4 | GUI editable priors | Planned |
| 6.5 | GUI loadable data | Planned |
| 6.6 | Refresh / propagate changes | Planned |

**Required dependencies (current):** `pyro-ppl >= 1.8`, `torch`, `matplotlib`, `scipy`, `networkx`, `pyyaml`, `dash >= 2.11`, `dash-cytoscape`.

**Additional dependencies for Phase 6:**
- `tkinter` (stdlib, for native file dialog in 6.3; may require `python3-tk` system package on Linux)
- `pandas` or `numpy` (for data file loading in 6.5, likely already present via torch)

**Note on Phase 6 architecture:** Features 6.1–6.6 all require the Dash app to manage mutable state across callbacks. The recommended pattern is a `dcc.Store(id="model-store", storage_type="memory")` component holding the serialized model state (graph topology, current prior expressions, current data), with all mutation callbacks reading from and writing back to this store. Inference should run in a background thread or process to avoid blocking the Dash server's event loop; `dcc.Interval` or `diskcache`-based long callbacks (`@app.long_callback`) handle the async signaling.
