# Ditto: Developer Instructions

This document provides step-by-step instructions for building Ditto from scratch. It assumes familiarity with Python but does not assume prior experience with Pyro or probabilistic programming.

---

## 1. Project Setup

### 1.1 Directory Structure

```
ditto/
    __init__.py
    parser.py
    graph_builder.py
    inference_engine.py
    visualizer.py
    cli.py
    models.py          # shared data structures (AnnotatedVariable, DittoGraph, InferenceResult)
tests/
    test_parser.py
    test_graph_builder.py
    test_inference_engine.py
    test_visualizer.py
    fixtures/
        simple_model.py
        linear_regression.py
ditto.yaml             # default configuration (optional; cli.py has built-in defaults)
pyproject.toml
```

### 1.2 Dependencies

```bash
pip install pyro-ppl torch matplotlib scipy networkx pyyaml dash dash-cytoscape
```

Specific package roles:
- `pyro-ppl >= 1.8`: probabilistic programming (sampling, SVI, `AutoNormal`, `AutoNormalizingFlow`, `Predictive`, `poutine`)
- `torch`: tensor operations; Pyro is built on PyTorch
- `matplotlib`: KDE and histogram plots, base64-encoded distribution images
- `scipy`: `scipy.stats.gaussian_kde` for kernel density estimation
- `networkx`: directed graph construction, layout, and validation
- `pyyaml`: YAML configuration file parsing (`yaml.safe_load` used in `cli.py`)
- `dash >= 2.11`: web application framework; `app.run()` (the non-deprecated call) requires >= 2.11
- `dash-cytoscape`: `cyto.Cytoscape` with draggable nodes, `dagre` layout, and `mouseoverNodeData` callbacks

Standard library modules used: `ast`, `re`, `importlib`, `io`, `base64`, `dataclasses`, `uuid`.

Optional packages:
```bash
pip install pygraphviz  # for hierarchical layout in static --export path
```

### 1.3 Python Version

Python 3.9 or later is required. `ast.unparse` (used to reconstruct expression strings) was added in Python 3.9. Set `python_requires >= "3.9"` in `pyproject.toml`.

### 1.4 Configuration File

`cli.py` defines `DEFAULT_CONFIG` with all built-in defaults, so running `ditto model.py` without a `ditto.yaml` works out of the box. Create `ditto.yaml` to override any values:

```yaml
inference:
  svi_steps: 2000
  learning_rate: 0.01
  num_samples: 1000

visualization:
  kde_points: 200
  histogram_bins: 40
  figure_size: [4, 3]      # per-thumbnail figure size (inches)
  prior_color: "#4C72B0"
  observed_color: "#DD8452"
  latent_color: "#55A868"

server:
  port: 8050
  debug: false

export:
  path: null
  format: png
  dpi: 150
```

The `--port` CLI flag overrides `server.port`. The `export` keys are only used when `--export` is passed.

---

## 2. Implementing the Annotation Parser

The parser scans a Python source file, finds all `# !Ditto: <tag>` comments (including comma-separated multi-tags), and associates each with the next annotatable statement.

### 2.1 Define the Data Structures (`models.py`)

Define all shared data structures here so every other module can import from this file without creating import cycles.

```python
from dataclasses import dataclass, field
from typing import Dict, FrozenSet, List
import networkx as nx
import torch

@dataclass
class AnnotatedVariable:
    name: str             # Python variable name (LHS) or pyro site name (bare sample call)
    tag: str              # primary display tag: "prior" | "latent" | "observed"
    line_number: int      # 1-indexed line of the annotation comment
    raw_expression: str   # RHS expression via ast.unparse
    dependencies: List[str] = field(default_factory=list)   # filled by graph_builder
    is_sample_call: bool = False   # True when raw_expression is a (pyro.)sample(...) call
    tags: FrozenSet[str] = field(default_factory=frozenset) # all tags from the annotation

@dataclass
class DittoGraph:
    graph: nx.DiGraph
    variables: Dict[str, AnnotatedVariable]  # name -> variable

    def topological_order(self) -> List[str]:
        return list(nx.topological_sort(self.graph))

    def predecessors(self, name: str) -> List[str]:
        return list(self.graph.predecessors(name))

@dataclass
class InferenceResult:
    prior_samples: Dict[str, torch.Tensor]      # name -> [N, ...] prior draws
    posterior_samples: Dict[str, torch.Tensor]  # site name -> [N, ...] posterior draws
    losses: List[float]                          # ELBO per SVI step (empty = prior-only)
    num_samples: int

    @property
    def samples(self) -> Dict[str, torch.Tensor]:
        """Backward-compatible merged view. Posterior overrides prior per site."""
        merged = dict(self.prior_samples)
        merged.update(self.posterior_samples)
        return merged
```

The three valid tags are `"prior"`, `"latent"`, and `"observed"`. There is no `"approx"` tag; guides are created automatically by the inference engine.

### 2.2 The Two-Pass Strategy

**Pass 1 (regex, line-by-line):** Scan for `# !Ditto: <tag>` comments. The regex captures everything after the colon to support comma-separated multi-tags.

```python
import re

_ANNOTATION_RE = re.compile(r"#\s*!Ditto:\s*(.+)")

def find_annotation_lines(source: str) -> list[tuple[int, str]]:
    """Return [(line_number, raw_tag_string), ...] for all !Ditto annotations."""
    results = []
    for i, line in enumerate(source.splitlines(), start=1):
        match = _ANNOTATION_RE.search(line)
        if match:
            results.append((i, match.group(1).strip()))
    return results
```

**Multi-tag parsing:**

```python
VALID_TAGS = {"prior", "observed", "latent"}

def _parse_tags(raw: str) -> frozenset:
    """Split "prior, latent" into frozenset({"prior", "latent"})."""
    return frozenset(t.strip() for t in raw.split(","))

def _primary_tag(tags: frozenset) -> str:
    """Select a single display tag by priority: latent > observed > prior."""
    for t in ("latent", "observed", "prior"):
        if t in tags:
            return t
    return next(iter(tags))
```

**Pass 2 (AST):** Parse the entire file and find the closest annotatable statement after each comment. Three node types are handled: `ast.Assign`, `ast.AnnAssign`, and `ast.Expr` wrapping a `pyro.sample(...)` call.

```python
import ast

def _is_pyro_sample_call_ast(value_node: ast.AST) -> bool:
    if not isinstance(value_node, ast.Call):
        return False
    func = value_node.func
    if isinstance(func, ast.Name) and func.id == "sample":
        return True
    if isinstance(func, ast.Attribute) and func.attr == "sample":
        return True
    return False

def _find_assignment_after(tree, line: int):
    candidates = []
    for node in ast.walk(tree):
        if not hasattr(node, "lineno") or node.lineno <= line:
            continue
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            candidates.append(node)
        elif isinstance(node, ast.Expr) and _is_pyro_sample_call_ast(node.value):
            candidates.append(node)
    if not candidates:
        return None
    candidates.sort(key=lambda n: n.lineno)
    return candidates[0]
```

**Bare `pyro.sample(...)` handling:**

When the closest statement is an `ast.Expr(pyro.sample(...))`, Ditto uses the string-literal first argument as the variable name and sets `is_sample_call=True`:

```python
def _extract_sample_site_name(call_node: ast.Call, lineno: int) -> str:
    if not call_node.args:
        raise ValueError(f"pyro.sample(...) at line {lineno} requires a site name.")
    first = call_node.args[0]
    if not isinstance(first, ast.Constant) or not isinstance(first.value, str):
        raise ValueError(
            f"First argument to pyro.sample must be a string literal at line {lineno}."
        )
    return first.value
```

### 2.3 Assembling `parse_file`

```python
from ditto.models import AnnotatedVariable

def parse_file(filepath: str) -> list[AnnotatedVariable]:
    with open(filepath, "r", encoding="utf-8") as f:
        source = f.read()

    tree = ast.parse(source, filename=filepath)
    annotations = find_annotation_lines(source)
    variables = []
    seen_names = set()

    for line_no, raw_tag in annotations:
        tags = _parse_tags(raw_tag)
        unknown = tags - VALID_TAGS
        if unknown:
            raise ValueError(f"Unknown tag '{next(iter(unknown))}' at line {line_no}.")
        tag = _primary_tag(tags)

        assign_node = _find_assignment_after(tree, line_no)
        if assign_node is None:
            raise ValueError(
                f"Annotation '{tag}' at line {line_no} not followed by an assignment."
            )

        if isinstance(assign_node, ast.Expr):
            # Bare pyro.sample(...) statement
            name = _extract_sample_site_name(assign_node.value, assign_node.lineno)
            raw_expression = ast.unparse(assign_node.value)
            is_sample_call = True
        else:
            name = _extract_target_name(assign_node)   # raises ValueError for non-Name targets
            raw_expression = ast.unparse(assign_node.value)
            is_sample_call = _is_pyro_sample_call_ast(assign_node.value)

        if name in seen_names:
            raise ValueError(f"Duplicate annotated variable name '{name}'.")
        seen_names.add(name)

        variables.append(AnnotatedVariable(
            name=name, tag=tag, line_number=line_no,
            raw_expression=raw_expression,
            is_sample_call=is_sample_call, tags=tags,
        ))

    return variables
```

---

## 3. Building the Dependency Graph

The graph builder infers directed edges from variable usage and resolves transitive dependencies through unannotated intermediate assignments.

### 3.1 Name Extraction via AST

```python
import ast

def extract_names(expression: str) -> set[str]:
    """Return all bare ast.Name identifiers in an expression string."""
    tree = ast.parse(expression, mode="eval")
    return {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
```

### 3.2 Collecting Unannotated Intermediates

When `filepath` is provided, all unannotated assignments in the file are collected so transitive dependencies can be resolved:

```python
def _collect_intermediates(filepath: str) -> dict[str, set[str]]:
    with open(filepath, "r", encoding="utf-8") as fh:
        source = fh.read()
    tree = ast.parse(source)
    intermediates: dict[str, set[str]] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            if len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
                name = node.targets[0].id
                refs = extract_names(ast.unparse(node.value))
                intermediates[name] = intermediates.get(name, set()) | refs
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            if node.value is not None:
                name = node.target.id
                refs = extract_names(ast.unparse(node.value))
                intermediates[name] = intermediates.get(name, set()) | refs
    return intermediates
```

### 3.3 Transitive Dependency Resolution

```python
def _resolve_deps(names, known_names, intermediates, visited=None) -> set[str]:
    if visited is None:
        visited = set()
    result = set()
    for name in names:
        if name in visited:
            continue
        visited.add(name)
        if name in known_names:
            result.add(name)
        elif name in intermediates:
            result |= _resolve_deps(intermediates[name], known_names, intermediates, visited)
    return result
```

If a name is annotated, it is added directly. If it is an unannotated intermediate, the search recurses into that variable's own references. This ensures a chain `stress → mu_k (unannotated) → knowledge` produces the edge `stress → knowledge`.

### 3.4 Building the Graph

```python
import networkx as nx
from ditto.models import AnnotatedVariable, DittoGraph

def build_graph(
    variables: list[AnnotatedVariable],
    filepath: str | None = None,
) -> DittoGraph:
    known_names = {v.name for v in variables}
    intermediates = _collect_intermediates(filepath) if filepath else {}
    # Annotated names must not be treated as pass-throughs
    for name in known_names:
        intermediates.pop(name, None)

    graph = nx.DiGraph()
    var_map = {}
    for v in variables:
        graph.add_node(v.name, tag=v.tag)
        var_map[v.name] = v

    for v in variables:
        referenced = extract_names(v.raw_expression)
        deps = _resolve_deps(referenced, known_names, intermediates) - {v.name}
        v.dependencies = sorted(deps)
        for dep in deps:
            graph.add_edge(dep, v.name)

    if not nx.is_directed_acyclic_graph(graph):
        cycle = nx.find_cycle(graph)
        raise ValueError(f"Dependency cycle detected: {cycle}")

    return DittoGraph(graph=graph, variables=var_map)
```

Pass `filepath=args.filepath` from the CLI so transitive resolution is active in normal usage.

---

## 4. Running SVI and Collecting Samples

The inference engine never requires the user to declare a guide variable. It selects and constructs the guide automatically based on how latent variables are annotated.

### 4.1 Preparing the Evaluation Namespace

```python
import torch, pyro
import pyro.distributions as dist

EVAL_NAMESPACE = {"torch": torch, "pyro": pyro, "dist": dist}
```

### 4.2 Sampling Prior Variables

```python
from ditto.models import AnnotatedVariable

def sample_prior(variable: AnnotatedVariable, num_samples: int) -> torch.Tensor:
    # When the expression is a pyro.sample(...) call, extract the distribution argument
    if variable.is_sample_call or is_pyro_sample_call(variable.raw_expression):
        expression_to_eval = extract_distribution_from_sample_call(variable.raw_expression)
    else:
        expression_to_eval = variable.raw_expression

    distribution = eval(expression_to_eval, dict(EVAL_NAMESPACE))
    if not hasattr(distribution, "sample"):
        raise TypeError(
            f"Prior expression for '{variable.name}' did not evaluate to a distribution "
            f"(got {type(distribution).__name__})."
        )
    return distribution.sample((num_samples,)).detach()
```

**`extract_distribution_from_sample_call`** extracts the second positional argument of a `pyro.sample(name, dist, ...)` call:

```python
def extract_distribution_from_sample_call(expression: str) -> str:
    tree = ast.parse(expression, mode="eval")
    node = tree.body
    if not isinstance(node, ast.Call) or len(node.args) < 2:
        raise ValueError(f"Expected pyro.sample(name, dist, ...), got: {expression!r}")
    return ast.unparse(node.args[1])
```

### 4.3 Prior Predictive for `pyro.sample(...)` Latents

When a latent variable's expression is a `pyro.sample(...)` call, it may reference variables that only exist inside the model's execution scope. These cannot be eval'd in isolation. Instead, run a prior predictive pass:

```python
from pyro.infer import Predictive

prior_predictive = Predictive(user_module.model, num_samples=num_samples)
prior_draws = prior_predictive(*pp_args, **pp_kwargs)
# Extract by pyro site name (the string-literal first arg of the sample call)
for var in sample_call_prior_latents:
    site_name = _get_pyro_site_name(var.raw_expression)
    tensor = prior_draws.get(site_name)
    if tensor is not None:
        prior_samples[var.name] = tensor.detach()
```

### 4.4 Auto-Guide Selection

The guide is created automatically — no user annotation is needed:

```python
from pyro.infer.autoguide import AutoNormal, AutoNormalizingFlow
from pyro.distributions.transforms import block_autoregressive
from pyro import poutine

# Block discrete sites from the autoguide (they can't be handled by continuous guides)
discrete_sites = find_discrete_latent_sites(model, args, kwargs)
guide_model = (
    poutine.block(model, hide=list(discrete_sites)) if discrete_sites else model
)

# AutoNormalizingFlow for pyro.sample(...) latents; AutoNormal otherwise
any_sample_call_latent = any(
    v.is_sample_call or is_pyro_sample_call(v.raw_expression)
    for v in latent_vars
)
if any_sample_call_latent:
    guide = AutoNormalizingFlow(guide_model, block_autoregressive)
else:
    guide = AutoNormal(guide_model)
```

`find_discrete_latent_sites` runs the model once under a Pyro trace and returns site names whose distribution has `has_enumerate_support = True`.

### 4.5 Running SVI

Use `pyro.optim.Adam` — **not** `torch.optim.Adam`. Pyro's `SVI` requires a Pyro-style optimizer (a callable that constructs per-parameter optimizers). `torch.optim.Adam` will raise an error at SVI construction time.

```python
from pyro.infer import SVI, Trace_ELBO

def run_svi(model, guide, model_args, model_kwargs, svi_steps, learning_rate) -> list[float]:
    pyro.clear_param_store()   # always clear before each run
    optimizer = pyro.optim.Adam({"lr": learning_rate})
    svi = SVI(model, guide, optimizer, loss=Trace_ELBO())

    losses = []
    for step in range(svi_steps):
        loss = svi.step(*model_args, **model_kwargs)
        losses.append(float(loss))
        if step % 200 == 0:
            print(f"[SVI] step {step:>5d}  loss = {loss:.4f}")
    return losses
```

### 4.6 Posterior Predictive Sampling

`poutine.uncondition` strips all `obs=...` conditioning from the model, converting observed sites into fresh sample sites. This is more robust than removing kwargs:

```python
from pyro.infer import Predictive
from pyro import poutine

def collect_posterior_samples(model, guide, model_args, model_kwargs, num_samples):
    unconditioned = poutine.uncondition(model)
    predictive = Predictive(unconditioned, guide=guide, num_samples=num_samples)
    raw = predictive(*model_args, **model_kwargs)
    return {name: tensor.detach() for name, tensor in raw.items()}
```

### 4.7 Surfacing Observed Data

Actual observed values are concatenated onto the posterior predictive draws for `observed`-tagged sites so the visualizer can render both in a single KDE:

```python
observed_values = extract_observed_data(model, args, kwargs)
for site_name, obs_tensor in observed_values.items():
    if site_name not in observed_var_names:
        continue
    existing = posterior_samples.get(site_name)
    if existing is None:
        posterior_samples[site_name] = obs_tensor
    else:
        obs_expanded = obs_tensor.unsqueeze(0)
        try:
            posterior_samples[site_name] = torch.cat([existing, obs_expanded], dim=0)
        except RuntimeError:
            pass  # shape mismatch — keep predictive draws alone
```

`extract_observed_data` traces the model once and returns values for every site flagged `is_observed`.

### 4.8 `get_model_args` — Providing Observed Data

Users must define a `get_data()` function in their model file:

```python
def get_data():
    x = torch.tensor([1.0, 2.0, 3.0])
    y = torch.tensor([2.1, 3.9, 6.2])
    return (x,), {"obs": y}
```

`get_model_args` calls this function and raises a helpful `AttributeError` if it is missing.

### 4.9 Top-Level `run_inference`

```python
from ditto.models import DittoGraph, InferenceResult

def run_inference(ditto_graph: DittoGraph, user_module, config: dict) -> InferenceResult:
    num_samples = config["inference"]["num_samples"]
    prior_samples, posterior_samples = {}, {}

    # Collect prior draws (eval in isolation for bare-dist latents/priors;
    # prior predictive pass for pyro.sample(...) latents)
    sample_call_prior_latents = []
    for name in ditto_graph.topological_order():
        var = ditto_graph.variables[name]
        if var.tag in ("prior", "latent"):
            if var.is_sample_call or is_pyro_sample_call(var.raw_expression):
                sample_call_prior_latents.append(var)
            else:
                prior_samples[name] = sample_prior(var, num_samples)

    if sample_call_prior_latents:
        pp_args, pp_kwargs = get_model_args(user_module, config)
        prior_draws = Predictive(user_module.model, num_samples=num_samples)(*pp_args, **pp_kwargs)
        for var in sample_call_prior_latents:
            site_name = _get_pyro_site_name(var.raw_expression)
            tensor = prior_draws.get(site_name)
            if tensor is not None:
                prior_samples[var.name] = tensor.detach()

    losses = []
    latent_vars = [v for v in ditto_graph.variables.values() if v.tag == "latent"]
    if latent_vars:
        # Auto-create guide, run SVI, collect posterior
        ...   # (see sections 4.4 – 4.7 above)

    return InferenceResult(
        prior_samples=prior_samples,
        posterior_samples=posterior_samples,
        losses=losses,
        num_samples=num_samples,
    )
```

---

## 5. Plotting KDE/Histograms per Variable

### 5.1 Choosing KDE vs. Histogram

Use KDE for continuous variables. Fall back to histogram when the number of unique sample values is less than 10, or when KDE raises `numpy.linalg.LinAlgError` (zero-variance input):

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

def plot_variable(samples, variable, ax, color, kde_points=200, histogram_bins=40, title=None):
    arr = samples.detach().cpu().numpy().reshape(-1)
    plot_title = title if title is not None else f"{variable.name} ({variable.tag})"

    if arr.size == 0:
        ax.set_title(plot_title)
        ax.text(0.5, 0.5, "no samples", ha="center", va="center", transform=ax.transAxes)
        return

    if len(np.unique(arr)) < 10:
        ax.hist(arr, bins=histogram_bins, color=color)
        ax.set_title(plot_title)
        return

    try:
        kde = stats.gaussian_kde(arr, bw_method="silverman")
        xs = np.linspace(arr.min(), arr.max(), kde_points)
        ys = kde(xs)
        ax.plot(xs, ys, color=color)
        ax.fill_between(xs, ys, alpha=0.3, color=color)
    except np.linalg.LinAlgError:
        ax.hist(arr, bins=histogram_bins, color=color)
    ax.set_title(plot_title)
```

### 5.2 Rendering to Base64

```python
import base64, io

def plot_variable_to_base64(samples, variable, config, title=None, color=None) -> str:
    """Returns the raw base64 string with no data:image/png;base64, prefix."""
    viz_cfg = config.get("visualization", {})
    figure_size = tuple(viz_cfg.get("figure_size", (4, 3)))
    kde_points = viz_cfg.get("kde_points", 200)
    histogram_bins = viz_cfg.get("histogram_bins", 40)
    plot_color = color or TAG_COLORS.get(variable.tag, "#888888")

    fig, ax = plt.subplots(figsize=figure_size)
    try:
        plot_variable(samples, variable, ax, color=plot_color,
                      kde_points=kde_points, histogram_bins=histogram_bins, title=title)
        buf = io.BytesIO()
        fig.tight_layout()
        fig.savefig(buf, format="png", dpi=100)
        buf.seek(0)
        return base64.b64encode(buf.read()).decode("utf-8")
    finally:
        plt.close(fig)
```

This function is called once per node at app startup; the Dash hover callback only does a dict lookup.

---

## 6. Rendering the DAG as an Interactive Dash App

### 6.1 Converting the Graph to Cytoscape Elements

Each node carries up to two base64 images: `image_prior` and `image_posterior`. `latent` nodes get both; `prior` nodes get only `image_prior`; `observed` nodes get only `image_posterior`.

```python
TAG_COLORS = {"prior": "#4C72B0", "observed": "#DD8452", "latent": "#55A868"}
prior_color = TAG_COLORS["prior"]
posterior_color = TAG_COLORS["observed"]

def build_cytoscape_elements(ditto_graph, inference_result, config) -> list[dict]:
    elements = []
    for name, variable in ditto_graph.variables.items():
        prior_b64 = ""
        posterior_b64 = ""

        prior_tensor = inference_result.prior_samples.get(name)
        posterior_tensor = inference_result.posterior_samples.get(name)

        show_prior = "prior" in variable.tags
        show_posterior = bool(variable.tags & {"latent", "observed"})

        if show_prior and prior_tensor is not None:
            prior_b64 = plot_variable_to_base64(prior_tensor, variable, config,
                                                title=f"{name} — Prior", color=prior_color)
        if show_posterior and posterior_tensor is not None:
            lbl = "Posterior" if "latent" in variable.tags else "posterior predictive"
            posterior_b64 = plot_variable_to_base64(posterior_tensor, variable, config,
                                                    title=f"{name} — {lbl}",
                                                    color=posterior_color)

        primary = prior_b64 if prior_b64 else posterior_b64
        elements.append({"data": {
            "id": name, "label": name, "tag": variable.tag,
            "tags": sorted(variable.tags),
            "image": primary, "image_prior": prior_b64, "image_posterior": posterior_b64,
        }})

    for u, v in ditto_graph.graph.edges():
        elements.append({"data": {"id": f"{u}__{v}", "source": u, "target": v}})

    return elements
```

### 6.2 Stylesheet

```python
def _build_stylesheet(config) -> list[dict]:
    viz_cfg = config.get("visualization", {})
    prior_color = viz_cfg.get("prior_color", TAG_COLORS["prior"])
    observed_color = viz_cfg.get("observed_color", TAG_COLORS["observed"])
    latent_color = viz_cfg.get("latent_color", TAG_COLORS["latent"])
    return [
        {"selector": "node", "style": {
            "label": "data(label)", "text-valign": "center", "text-halign": "center",
            "color": "white", "font-size": "14px", "width": 60, "height": 60,
        }},
        {"selector": 'node[tag = "prior"]', "style": {"background-color": prior_color}},
        {"selector": 'node[tag = "observed"]', "style": {"background-color": observed_color}},
        {"selector": 'node[tag = "latent"]', "style": {"background-color": latent_color}},
        {"selector": "edge", "style": {
            "curve-style": "bezier", "target-arrow-shape": "triangle",
            "line-color": "#999", "target-arrow-color": "#999", "width": 2,
        }},
    ]
```

### 6.3 `create_dash_app`

`cyto.load_extra_layouts()` must be called once at module import time to register the `dagre` layout. Without it, Cytoscape.js silently falls back to `circle` layout:

```python
import dash, dash_cytoscape as cyto
cyto.load_extra_layouts()   # module-level call

from dash import html, Input, Output

def create_dash_app(ditto_graph, inference_result, config) -> dash.Dash:
    elements = build_cytoscape_elements(ditto_graph, inference_result, config)
    stylesheet = _build_stylesheet(config)

    prior_lookup = {}
    posterior_lookup = {}
    for elem in elements:
        if "source" in elem["data"]:
            continue
        node_id = elem["data"]["id"]
        prior_lookup[node_id] = elem["data"].get("image_prior", "")
        posterior_lookup[node_id] = elem["data"].get("image_posterior", "")

    app = dash.Dash(__name__)
    app.title = "Ditto"

    app.layout = html.Div(
        style={"display": "flex", "flexDirection": "row", "height": "100vh"},
        children=[
            html.Div(
                style={"flex": "3", "borderRight": "1px solid #ddd"},
                children=[
                    html.H2("Ditto: Pyro DAG Visualizer", style={"padding": "8px"}),
                    cyto.Cytoscape(
                        id="cytoscape",
                        layout={"name": "dagre"},
                        autoRefreshLayout=False,
                        userPanningEnabled=True,
                        userZoomingEnabled=True,
                        style={"width": "100%", "height": "90%"},
                        elements=elements,
                        stylesheet=stylesheet,
                    ),
                ],
            ),
            html.Div(
                id="tooltip-div",
                style={"flex": "1", "padding": "12px", "overflowY": "auto"},
                children=html.Div("Hover over a node to see its distribution."),
            ),
        ],
    )

    @app.callback(
        Output("tooltip-div", "children"),
        Input("cytoscape", "mouseoverNodeData"),
    )
    def _show_node(node_data):
        if not node_data:
            return html.Div("Hover over a node to see its distribution.")
        name = node_data.get("id", "")
        tag = node_data.get("tag", "")
        tags = set(node_data.get("tags", [tag]))
        prior_b64 = prior_lookup.get(name, "")
        posterior_b64 = posterior_lookup.get(name, "")

        if not prior_b64 and not posterior_b64:
            return html.Div([html.H3(f"{name} ({tag})"),
                             html.P("No samples available.")])

        children = [html.H3(f"{name} ({tag})")]
        if prior_b64:
            if "latent" in tags or "observed" in tags:
                children.append(html.H4("Prior"))
            children.append(html.Img(src=f"data:image/png;base64,{prior_b64}",
                                     style={"maxWidth": "100%"}))
        if posterior_b64:
            if "prior" in tags:
                children.append(html.H4("Posterior"))
            children.append(html.Img(src=f"data:image/png;base64,{posterior_b64}",
                                     style={"maxWidth": "100%"}))
        return html.Div(children)

    return app
```

`autoRefreshLayout=False` ensures that user-repositioned nodes are not reset on each callback.

---

## 7. CLI Integration

### 7.1 Built-in Defaults and Config Loading

`cli.py` defines `DEFAULT_CONFIG` so the tool works without a `ditto.yaml`. `_deep_merge` layers the user's YAML on top of the defaults so partial configs don't leave keys missing:

```python
DEFAULT_CONFIG = {
    "inference": {"svi_steps": 2000, "learning_rate": 0.01, "num_samples": 1000},
    "visualization": {"kde_points": 200, "histogram_bins": 40, "figure_size": [4, 3],
                      "prior_color": "#4C72B0", "observed_color": "#DD8452",
                      "latent_color": "#55A868"},
    "server": {"port": 8050, "debug": False},
    "export": {"path": None, "format": "png", "dpi": 150},
}

def _deep_merge(base: dict, override: dict) -> dict:
    out = dict(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out

def load_config(config_path: str) -> dict:
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            user_cfg = yaml.safe_load(f) or {}
    except FileNotFoundError:
        return dict(DEFAULT_CONFIG)
    return _deep_merge(DEFAULT_CONFIG, user_cfg)
```

### 7.2 `main`

```python
def main(argv=None) -> int:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)

    try:
        from ditto.parser import parse_file
        from ditto.graph_builder import build_graph
        from ditto.inference_engine import load_user_module, run_inference
        from ditto.visualizer import create_dash_app, render_dag_static

        config = load_config(args.config)
        variables = parse_file(args.filepath)
        # Pass filepath so graph_builder can resolve transitive deps
        ditto_graph = build_graph(variables, filepath=args.filepath)
        user_module = load_user_module(args.filepath)
        inference_result = run_inference(ditto_graph, user_module, config)

        if args.export:
            render_dag_static(ditto_graph, inference_result, config, args.export)

        if args.dry_run:
            return 0

        app = create_dash_app(ditto_graph, inference_result, config)
        port = args.port if args.port is not None else config["server"]["port"]
        app.run(debug=False, port=port)  # blocking; app.run_server is deprecated in Dash >= 2.11
        return 0
    except Exception as exc:
        import traceback
        print(f"[ditto] error: {exc}", file=sys.stderr)
        traceback.print_exc()
        return 1
```

Access config values with their nested structure (`config["inference"]["svi_steps"]`) to avoid key collisions between sections.

---

## 8. Testing

Write tests in `tests/` and run with `pytest`.

**Parser tests** (`tests/test_parser.py`): create small in-memory Python strings with known annotations and assert the correct `AnnotatedVariable` objects (including `tags`, `is_sample_call`, and `tag`) are returned. Test multi-tag annotations (`"prior, latent"`), bare `pyro.sample(...)` statements, and duplicate-name detection.

**Graph builder tests** (`tests/test_graph_builder.py`): construct `AnnotatedVariable` lists manually and assert that edges are created for known dependencies. Test transitive resolution by passing a `filepath` to a fixture file that contains unannotated intermediates.

**Inference tests** (`tests/test_inference_engine.py`): use a simple Bayesian linear regression fixture. Run with a small number of SVI steps (e.g., 50) for speed. Assert that `prior_samples` contains entries for `prior`/`latent` variables, `posterior_samples` contains entries for `latent`/`observed` sites, and all tensors have the correct leading `num_samples` dimension.

**Visualizer tests** (`tests/test_visualizer.py`): create dummy sample tensors and call `plot_variable`; assert no exception is raised. For `plot_variable_to_base64`, assert the return value is a non-empty string. For `build_cytoscape_elements`, assert the list has the correct node and edge count and each node dict contains `id`, `label`, `tag`, `image_prior`, and `image_posterior` keys.

```bash
pytest tests/ -v
```

---

## 9. Annotating Your Model File

The three valid tags are `prior`, `latent`, and `observed`. Multi-tag annotations are supported.

**`prior`**: a variable sampled from its prior distribution; no posterior is computed. Only a prior plot is shown in the hover panel.

**`latent`**: a latent random variable. Ditto samples its prior for the left plot and uses SVI + `Predictive` to compute its posterior for the right plot. Annotate as `# !Ditto: prior, latent` to indicate that the prior distribution should also be displayed separately.

**`observed`**: a conditioned observation site. Ditto shows the posterior predictive distribution (plus the actual observed data concatenated in) in the hover panel.

**Example — variables as assignments:**
```python
# !Ditto: prior
mu = dist.Normal(0., 5.)

# !Ditto: prior
sigma = dist.HalfNormal(1.)

# !Ditto: observed
obs = dist.Normal(mu, sigma)
```

**Example — variables as bare `pyro.sample(...)` calls inside a model function:**
```python
def model(x, obs=None):
    # !Ditto: latent
    alpha = pyro.sample("alpha", dist.Normal(0., 1.))

    # !Ditto: latent
    beta = pyro.sample("beta", dist.Normal(0., 1.))

    # !Ditto: observed
    pyro.sample("obs", dist.Normal(alpha + beta * x, 0.1), obs=obs)
```

In both cases, the user must also define:

```python
def get_data():
    return (x_tensor,), {"obs": y_tensor}
```

No guide variable is needed. Ditto creates the guide automatically.
