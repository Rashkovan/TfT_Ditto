# Ditto Software Architecture

This document describes the internal architecture of Ditto: a DAG visualization tool for Pyro probabilistic models. It covers module responsibilities, data flow, key data structures, design decisions, and extension points.

---

## 1. High-Level Data Flow

```
Source file (.py)  +  ditto.yaml config  +  user get_data()
    |
    v
[parser.py]
    -- regex scan for # !Ditto: comments  (regex is NECESSARY; AST has no comment nodes)
    -- supports comma-separated multi-tags: "# !Ditto: prior, latent"
    -- AST parse for assignment targets and expressions
    -- also handles bare pyro.sample(...) expression statements (site name from string literal)
    |
    v
List[AnnotatedVariable]
    |
    v
[graph_builder.py]
    -- AST-based name extraction from expressions
    -- transitive dep resolution through unannotated intermediates (when filepath provided)
    -- Edge inference: dep_name --> variable_name
    -- Acyclicity validation
    |
    v
DittoGraph (networkx.DiGraph + metadata)
    |
    v
[inference_engine.py]
    -- prior/latent: eval distribution expression, draw samples directly
    -- latent (pyro.sample call): prior predictive via Predictive(model, num_samples=N)
    -- latent present: auto-create AutoNormal or AutoNormalizingFlow guide over user model
    -- run SVI (pyro.optim.Adam + Trace_ELBO); pyro.clear_param_store() before each run
    -- posterior: Predictive over trained guide with poutine.uncondition for all obs sites
    -- discrete latent sites blocked from autoguide via poutine.block
    |
    v
InferenceResult (prior_samples + posterior_samples, all in constrained space)
    |
    v
[visualizer.py]
    -- per-variable KDE / histogram plots (pre-computed as base64 PNG strings at startup)
    -- latent nodes get both a prior and posterior plot
    -- DAG converted to Cytoscape element dicts (nodes + edges)
    -- Dash app assembled with cyto.Cytoscape component and hover callback
    |
    v
Dash web app (localhost:port)    [primary output]
Static PNG/SVG via --export      [optional]
```

---

## 2. Module Breakdown

### 2.1 `models.py` — Shared Data Structures

All cross-module data structures are defined here to avoid circular imports and to serve as a single source of truth for the shape of data flowing through the pipeline.

**Responsibilities:**
- Define `AnnotatedVariable`, `DittoGraph`, and `InferenceResult`.
- Provide no business logic; this is a pure data-definition module.

**Key contents:**

```python
from dataclasses import dataclass, field
from typing import Dict, FrozenSet, List
import networkx as nx
import torch

@dataclass
class AnnotatedVariable:
    name: str             # left-hand side identifier (or pyro site name for bare sample calls)
    tag: str              # primary display tag: "prior" | "latent" | "observed"
    line_number: int      # line of the annotation comment in the source file
    raw_expression: str   # right-hand side reconstructed from AST (ast.unparse)
    dependencies: List[str] = field(default_factory=list)
    # populated by graph_builder after edge inference
    is_sample_call: bool = False
    # True when raw_expression is a (pyro.)sample(...) call — inference engine
    # extracts the underlying distribution rather than eval-ing the whole call
    tags: FrozenSet[str] = field(default_factory=frozenset)
    # full set of tags from the annotation, may include multiple (e.g. {"prior","latent"})

@dataclass
class DittoGraph:
    graph: nx.DiGraph                        # nodes are variable names
    variables: Dict[str, AnnotatedVariable]  # name -> metadata

    def topological_order(self) -> List[str]: ...
    def predecessors(self, name: str) -> List[str]: ...

@dataclass
class InferenceResult:
    prior_samples: Dict[str, torch.Tensor]     # variable name -> [N, ...] prior draws
    posterior_samples: Dict[str, torch.Tensor] # site name -> [N, ...] posterior draws
    losses: List[float]                        # ELBO loss curve (empty if no SVI ran)
    num_samples: int

    @property
    def samples(self) -> Dict[str, torch.Tensor]:
        # Backward-compatible merged view: posterior overrides prior per site.
        ...
```

The three-tag vocabulary is `"prior"`, `"latent"`, and `"observed"`. The old `"approx"` tag no longer exists; guides are constructed automatically by the inference engine.

---

### 2.2 `parser.py` — Annotation Parser

**Responsibilities:**
- Locate `# !Ditto: <tag>` comments in source files. Tags may be comma-separated (e.g., `# !Ditto: prior, latent`).
- Associate each comment with the immediately following annotatable statement.
- Handle two statement forms: `ast.Assign` / `ast.AnnAssign` (name extracted from LHS) and bare `ast.Expr` wrapping a `(pyro.)sample(...)` call (name extracted from the string-literal first argument).
- Return a validated list of `AnnotatedVariable` objects.

**Key functions:**
- `find_annotation_lines(source: str) -> List[Tuple[int, str]]`
- `_parse_tags(raw: str) -> frozenset` — splits comma-separated tag strings
- `_primary_tag(tags: frozenset) -> str` — selects display tag: `latent > observed > prior`
- `parse_file(filepath: str) -> List[AnnotatedVariable]`

**Internal design:**
The parser uses a two-pass approach. Pass 1 uses a compiled regex to locate annotation comment lines. This is necessary — not merely faster — because Python's `ast` module does not include comment nodes in the parse tree at any Python version. Pass 2 walks the AST to find the closest annotatable statement after each comment's line number.

Multi-tag annotations (`# !Ditto: prior, latent`) are parsed by `_parse_tags`, which splits on commas. `_primary_tag` establishes a priority order (`latent > observed > prior`) for the single display tag stored in `AnnotatedVariable.tag`, used for CSS selectors and node colors.

Bare `pyro.sample(...)` expression statements are handled as a third case in the AST walk: the site name (string-literal first argument) is used as the variable name, and `is_sample_call` is set to `True` so the inference engine knows to extract the underlying distribution rather than eval-ing the whole call.

---

### 2.3 `graph_builder.py` — Dependency Graph Constructor

**Responsibilities:**
- Accept a list of `AnnotatedVariable` objects and an optional `filepath`.
- Infer directed dependency edges from variable name references in expressions.
- Transitively resolve dependencies through unannotated intermediate assignments when `filepath` is provided.
- Construct and return a `DittoGraph`.
- Validate acyclicity.

**Key functions:**
- `extract_names(expression: str) -> Set[str]`
- `_collect_intermediates(filepath: str) -> Dict[str, Set[str]]`
- `_resolve_deps(names, known_names, intermediates, visited) -> Set[str]`
- `build_graph(variables: List[AnnotatedVariable], filepath: Optional[str] = None) -> DittoGraph`

**Internal design:**
When `filepath` is provided, `_collect_intermediates` parses the entire source file and builds a mapping from every unannotated assignment name to the set of names it references. `_resolve_deps` then walks this mapping transitively: if a referenced name is an annotated variable it is added to the dependency set; if it is an unannotated intermediate, the search recurses into that intermediate's references. This ensures that a chain like `stress → mu_k → knowledge` (where `mu_k` is unannotated) produces a direct edge `stress → knowledge`.

Cycles in unannotated intermediates are broken by a `visited` guard. Annotated variable names are stripped from the intermediates map before resolution so they always terminate the search rather than being treated as pass-through nodes.

The set of references is intersected with the set of known annotated variable names (possibly via the transitive resolution above) to produce the dependency set.

---

### 2.4 `inference_engine.py` — Inference and Sampling

**Responsibilities:**
- Sample `prior` and `latent` variables from their declared distribution expressions.
- For `latent` variables annotated via bare `pyro.sample(...)` calls, run a prior predictive pass (`Predictive(model, num_samples=N)`) rather than eval-ing in isolation.
- Auto-create a guide over the user's `model` callable when any `latent` variables are present: `AutoNormal` when all latent expressions are bare distributions; `AutoNormalizingFlow` (with `block_autoregressive`) when any latent is a `pyro.sample(...)` call.
- Block discrete latent sites from the autoguide via `poutine.block` before guide construction.
- Call `pyro.clear_param_store()` at the start of each SVI run.
- Collect posterior samples via `Predictive` wrapping the model in `poutine.uncondition` so all `obs=...` conditioning is stripped.
- Surface actual observed data alongside posterior predictive draws for `observed`-tagged sites.
- Return an `InferenceResult`.

**Key functions:**
- `is_pyro_sample_call(expression: str) -> bool`
- `extract_distribution_from_sample_call(expression: str) -> str`
- `load_user_module(filepath: str) -> ModuleType`
- `sample_prior(variable: AnnotatedVariable, num_samples: int) -> torch.Tensor`
- `run_svi(model, guide, model_args, model_kwargs, svi_steps, learning_rate) -> List[float]`
- `collect_posterior_samples(model, guide, model_args, model_kwargs, num_samples, observed_sites) -> Dict[str, Tensor]`
- `find_discrete_latent_sites(model, model_args, model_kwargs) -> set`
- `extract_observed_data(model, model_args, model_kwargs) -> Dict[str, Tensor]`
- `get_model_args(user_module, config) -> Tuple[tuple, dict]`
- `run_inference(ditto_graph: DittoGraph, user_module, config: dict) -> InferenceResult`

**Internal design:**
There is no user-declared guide variable. Ditto auto-creates the guide in `run_inference` based on whether any latent variable uses a `pyro.sample(...)` expression. This eliminates the `approx` tag entirely.

A unique module name is generated with `uuid.uuid4().hex` for each `load_user_module` call to avoid collisions in `sys.modules` during repeated runs (e.g., in tests).

`sample_prior` detects `is_sample_call` (or re-checks with `is_pyro_sample_call`) and calls `extract_distribution_from_sample_call` to obtain the second positional argument of the `pyro.sample(...)` call — the distribution sub-expression — before eval-ing it.

`collect_posterior_samples` wraps the model with `poutine.uncondition` before passing it to `Predictive`. This is more robust than stripping observed kwargs, because it works even when observed data is constructed inside the model body rather than passed as an argument.

`find_discrete_latent_sites` runs the model once under a Pyro trace and returns site names whose distribution has `has_enumerate_support = True`. These sites are passed to `poutine.block(model, hide=...)` so the autoguide only covers continuous latents.

After posterior sampling, `extract_observed_data` surfaces the actual observed values from the model trace. For each `observed`-tagged variable, the observed tensor is concatenated onto the posterior predictive draws along dim 0 so the visualizer can render both in a single KDE.

**`get_model_args` convention:** Users must define a top-level `get_data()` function returning `(args, kwargs)` to pass to `model`.

---

### 2.5 `visualizer.py` — Plot Renderer

**Responsibilities:**
- Render a KDE or histogram plot for each variable's samples.
- For `latent` nodes, pre-compute two thumbnails (prior and posterior) and display both in the hover panel.
- Convert the `DittoGraph` to a Cytoscape element format.
- Pre-compute base64-encoded distribution plot images per node at app startup.
- Assemble a Dash app with a `cyto.Cytoscape` component and a side-panel hover callback.
- Launch the Dash server (blocking call) or return the app object for programmatic use.

**Key functions:**
- `plot_variable(samples, variable, ax, color, kde_points, histogram_bins, title) -> None`
- `plot_variable_to_base64(samples, variable, config, title, color) -> str`
- `build_cytoscape_elements(ditto_graph, inference_result, config) -> list[dict]`
- `_build_stylesheet(config) -> list[dict]`
- `create_dash_app(ditto_graph, inference_result, config) -> dash.Dash`
- `render_dag_static(ditto_graph, inference_result, config, output_path) -> None`

**Internal design:**
Each node element in `build_cytoscape_elements` carries three image keys: `image_prior`, `image_posterior`, and `image` (the primary, for backward compatibility). For `prior` nodes only `image_prior` is populated; for `observed` nodes only `image_posterior`; for `latent` nodes both are populated. The hover callback reads `prior_lookup` and `posterior_lookup` dicts built from these keys and stacks the images vertically in the side panel.

The app layout is a flex row: the left column contains the `cyto.Cytoscape` component (3/4 width) and the right column (`id="tooltip-div"`) shows the hover content. This is a side-panel layout rather than a floating tooltip.

`TAG_COLORS` maps `"prior"`, `"observed"`, and `"latent"` to hex color strings. The `_build_stylesheet` function reads these from config (falling back to module-level defaults) and produces the Cytoscape stylesheet list.

`cyto.load_extra_layouts()` is called once at module import time to register the `dagre` layout algorithm. Without this call, Cytoscape.js silently falls back to `circle` layout.

KDE uses `scipy.stats.gaussian_kde` with `bw_method="silverman"`. `numpy.linalg.LinAlgError` (zero-variance input) is caught and falls back to a histogram. Variables with fewer than 10 unique values are always rendered as histograms.

---

### 2.6 `cli.py` — Command-Line Interface

**Responsibilities:**
- Define `DEFAULT_CONFIG` with all built-in defaults (avoids requiring a `ditto.yaml` to run).
- Deep-merge the user's YAML on top of defaults via `_deep_merge`.
- Parse command-line arguments.
- Orchestrate the pipeline by calling `parser`, `graph_builder`, `inference_engine`, and `visualizer` in sequence.
- Launch the Dash web server (`app.run(debug=False, port=port)`) as the final blocking step.

**CLI structure:** Flat `argparse` with a positional `filepath` argument and flags: `--config` (default `ditto.yaml`), `--port` (int, overrides config), `--export` (optional static PNG/SVG path), `--dry-run`.

**Key functions:**
- `load_config(config_path: str) -> dict`: loads `ditto.yaml` and deep-merges with `DEFAULT_CONFIG`.
- `_build_arg_parser() -> argparse.ArgumentParser`
- `main(argv) -> int`: orchestrates the full pipeline; returns the process exit code.

`main` passes `filepath=args.filepath` to `build_graph` so the graph builder can resolve transitive dependencies through unannotated intermediates in the source file.

`app.run()` is used instead of the deprecated `app.run_server()` (Dash >= 2.11).

---

## 3. Key Data Structures

### `AnnotatedVariable`

The primary unit produced by the parser and consumed by all downstream modules. The `tags` frozenset holds all tags from the annotation (e.g., `{"prior", "latent"}` for `# !Ditto: prior, latent`). The `tag` string is the single primary/display tag derived by `_primary_tag`. `is_sample_call` flags expressions that are `pyro.sample(...)` calls rather than bare distributions — the inference engine uses this to extract the underlying distribution sub-expression before eval-ing.

### `DittoGraph`

A thin wrapper around a `networkx.DiGraph` that preserves the association between graph nodes (string variable names) and their full `AnnotatedVariable` metadata. Methods `topological_order` and `predecessors` delegate to networkx.

### `InferenceResult`

Stores prior and posterior samples in two separate dicts. `prior_samples` holds draws from the declared prior distribution for `prior` and `latent` variables. `posterior_samples` holds draws from the trained guide's posterior (via `Predictive`) for `latent` and `observed` sites. The `.samples` property merges them (posterior overrides prior) for backward-compatible access. `losses` is the per-step ELBO curve (empty for prior-only runs).

---

## 4. Design Decisions and Rationale

### 4.1 AST over Pure Regex for Parsing

Python source code is ambiguous to regex: the same pattern can appear in string literals, docstrings, f-strings, or commented-out code. The `ast` module parses Python unambiguously. Using AST for variable name and expression extraction prevents false positives.

The one exception is comment detection: Python's `ast` module intentionally omits comments. A focused regex scan for `# !Ditto:` comment lines is therefore unavoidable. The two-pass design uses each tool for what it is best at.

### 4.2 Auto-Guide Creation Instead of User-Declared Guide

Earlier designs required a user to annotate a guide variable with `# !Ditto: approx`. This was redundant: the only decision Ditto needs is whether to use `AutoNormal` (simpler, faster, no `pyro.sample(...)` latents) or `AutoNormalizingFlow` (more expressive, required when latent expressions are `pyro.sample(...)` calls inside the model). This choice can be made automatically by inspecting `is_sample_call` on the latent variables, so no user annotation is needed.

### 4.3 Dual Plots for Latent Variables

`latent` variables are interesting precisely because their posterior differs from their prior. Rendering both prior and posterior thumbnails in the hover panel lets the user visually compare the two distributions and assess whether SVI moved the posterior away from the prior — a direct diagnostic for inference quality.

### 4.4 `AutoNormalizingFlow` as the Default Guide for Sample-Call Latents

`AutoNormal` assumes posterior independence and unimodality. `AutoNormalizingFlow` with `block_autoregressive` can model complex, correlated posteriors at the cost of more SVI steps. Ditto selects `AutoNormalizingFlow` only when any latent variable is declared via `pyro.sample(...)`, because those variables live inside the model function and are more likely to have complex posterior geometry (they interact with plate loops, transformations, etc.).

**Important limitation:** Normalizing flows are diffeomorphisms from a unimodal base distribution. `AutoNormalizingFlow` cannot model multimodal posteriors. If the true posterior is multimodal, the flow will fit a unimodal approximation.

### 4.5 `poutine.uncondition` for Posterior Predictive Sampling

Rather than stripping observed kwargs, `collect_posterior_samples` wraps the model in `poutine.uncondition`. This converts every `obs=...` site into a fresh sample site, regardless of whether the data was passed as a kwarg or constructed inside the model body. It is more robust than trying to remove observed kwargs from the argument dict.

### 4.6 Transitive Dependency Resolution Through Unannotated Intermediates

Without transitive resolution, a chain `stress (annotated) → mu_k (unannotated) → knowledge (annotated)` would produce no edge in the DAG because `mu_k` is not in the annotated variable set. `graph_builder._collect_intermediates` collects every unannotated assignment in the source file, and `_resolve_deps` walks transitively through those intermediates to find the annotated variables they ultimately depend on. This produces a direct edge `stress → knowledge` reflecting the true dependency.

### 4.7 Eager Pre-computation of Base64 Images

Distribution plots are rendered to `io.BytesIO` buffers and base64-encoded once at app startup. The hover callback performs only a dict lookup — no matplotlib work occurs at interaction time. This keeps hover latency near zero regardless of model complexity.

### 4.8 `importlib.util` for User Module Loading

Using `importlib.util.spec_from_file_location` gives Ditto a proper module object with a namespace, allowing `getattr` lookups. A fresh UUID-based module name is used each time to avoid `sys.modules` cache collisions across repeated calls.

**Known constraint:** Relative imports in the user's model file will not work because `importlib.util.spec_from_file_location` for a standalone file sets `__package__` to `None`.

---

## 5. Extension Points

### 5.1 Adding New Tags

To add a new tag (e.g., `likelihood`):

1. Add the tag string to `VALID_TAGS` in `parser.py`.
2. Update `_primary_tag` priority order in `parser.py`.
3. Add a handling branch in `inference_engine.py`'s `run_inference`.
4. Add a visual style (color) in `visualizer.py`'s `TAG_COLORS` and `_build_stylesheet`.
5. Update `DEFAULT_CONFIG` in `cli.py` with any new config keys.

### 5.2 Swapping Inference Backends

`inference_engine.py` is the only module that depends on Pyro's SVI API. To swap in MCMC or NumPyro, replace `run_svi` and `collect_posterior_samples` with functions that accept the same signatures and return the same `Dict[str, torch.Tensor]` structure. The rest of the pipeline is backend-agnostic.

### 5.3 Interactive Features on Top of the Dash Baseline

The Dash + Cytoscape app is the primary output. Future interactive features can be layered on top without changing the pipeline:
- Click-to-expand: open a full-resolution distribution plot in a modal on node click.
- ELBO loss curve tab: a `dcc.Tab` displaying `InferenceResult.losses`.
- Variable detail panel: show tag, expression, and summary statistics alongside the plot.
- Editable priors: a form panel that allows editing the prior expression string and re-running inference.

### 5.4 Plugin Architecture for Custom Visualization Backends

To allow users to register custom plot types, define a `PlotBackend` abstract base class with a `plot(samples, variable, ax, **kwargs)` method and maintain a registry in `visualizer.py` mapping tag strings to `PlotBackend` subclasses.

---

## 6. Error Handling Strategy

Each module raises specific, descriptive exceptions:

- `parser.py`: raises `ValueError` for unknown tags or tags not followed by an assignment; raises `ValueError` for non-string-literal site names in bare `pyro.sample(...)` calls; raises `ValueError` for duplicate annotated variable names.
- `graph_builder.py`: raises `ValueError` with a descriptive cycle description if `nx.find_cycle` detects a cycle.
- `inference_engine.py`: raises `TypeError` if a prior expression does not evaluate to a distribution; raises `AttributeError` with a helpful message if `get_data` or `model` is missing from the user module; wraps Pyro SVI exceptions with context.
- `visualizer.py`: catches `numpy.linalg.LinAlgError` from KDE and falls back to a histogram.
- `cli.py`: catches all module-level exceptions, prints a user-friendly error with the exception type, and exits with a non-zero status code.

---

## 7. Module Dependency Graph

```
cli.py
  |-- parser.py          --> models.py (imports AnnotatedVariable)
  |-- graph_builder.py   --> models.py
  |-- inference_engine.py --> models.py
  |-- visualizer.py      --> models.py
  |
  models.py              (no internal deps)
```

`models.py` is a leaf with no dependencies on other Ditto modules. `parser.py` imports `AnnotatedVariable` from `models.py`. `graph_builder.py`, `inference_engine.py`, and `visualizer.py` each import from `models.py` but not from each other. `cli.py` is the only module that imports from all others. This flat, hub-and-spoke dependency structure prevents circular imports and makes each module independently testable.
