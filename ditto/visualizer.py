"""Render the DAG as an interactive Dash + Cytoscape application.

Per-node KDE/histogram thumbnails are pre-computed at startup and shipped to
the browser as base64-encoded PNGs embedded in the Cytoscape node data. A
hover callback then surfaces them in a tooltip ``<img>`` tag on demand.

For ``latent`` nodes, two thumbnails are pre-computed (prior and posterior)
and both are shown stacked vertically in the tooltip. ``prior`` and
``observed`` nodes show a single thumbnail.

Phase 6 extends the application with runtime-mutable state. A model loaded
at launch time still works exactly as it always did, but the Dash app now
also supports:

* Launching with no file at all (the canvas is empty until the user drops
  one in).
* Drag-and-drop loading of a ``.py`` file.
* A native "Open File…" button that uses ``tkinter.filedialog``.
* Click-to-edit prior expressions in the side panel.
* Loading data files (``.csv``, ``.npy``, ``.pkl``) to override
  ``get_data()``.
* A "Refresh" button that re-runs the full inference pipeline against the
  in-memory edits.

All shared mutable state lives in ``dcc.Store`` components so callbacks can
read and write it without rerunning inference unnecessarily.
"""
from __future__ import annotations

import ast
import base64
import io
import os
import pickle
import tempfile
import traceback
from typing import Dict, List, Optional, Tuple

import dash
import dash_cytoscape as cyto
import matplotlib

# Use a non-interactive backend; the app runs headless on a server and we
# never call ``plt.show()``.
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import networkx as nx  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from dash import Input, Output, State, dcc, html, no_update  # noqa: E402
from dash.exceptions import PreventUpdate  # noqa: E402
from scipy import stats  # noqa: E402

from ditto.models import AnnotatedVariable, DittoGraph, InferenceResult  # noqa: E402

# REQUIRED: dash-cytoscape's "dagre" layout (and other extras) live in a
# separately-loaded JS bundle. Without this call, asking for ``layout={"name":
# "dagre"}`` silently falls back to "circle" with no warning. Must run at
# module import time so it's in place before any Cytoscape component is built.
cyto.load_extra_layouts()


# Default visual styling for the three tag types.
TAG_COLORS = {
    "prior": "#4C72B0",
    "observed": "#DD8452",
    "latent": "#55A868",
}


def _to_numpy(samples: torch.Tensor) -> np.ndarray:
    """Flatten samples to a 1-D numpy array suitable for plotting."""
    arr = samples.detach().cpu().numpy()
    return arr.reshape(-1)


def plot_variable(
    samples: torch.Tensor,
    variable: AnnotatedVariable,
    ax,
    color: str,
    kde_points: int = 200,
    histogram_bins: int = 40,
    title: Optional[str] = None,
) -> None:
    """Plot a per-variable distribution onto an existing matplotlib ``ax``.

    Uses Gaussian KDE with Silverman's rule when there are enough unique
    values; otherwise falls back to a histogram. Also catches
    ``numpy.linalg.LinAlgError`` (which KDE raises on zero-variance inputs)
    and degrades gracefully.
    """
    arr = _to_numpy(samples)
    plot_title = title if title is not None else f"{variable.name} ({variable.tag})"

    if arr.size == 0:
        ax.set_title(plot_title)
        ax.text(0.5, 0.5, "no samples", ha="center", va="center",
                transform=ax.transAxes)
        return

    unique_vals = np.unique(arr)

    if len(unique_vals) < 10:
        ax.hist(arr, bins=min(histogram_bins, len(unique_vals)), color=color)
        ax.set_title(plot_title)
        return

    try:
        kde = stats.gaussian_kde(arr, bw_method="silverman")
        xs = np.linspace(arr.min(), arr.max(), kde_points)
        ys = kde(xs)
        ax.plot(xs, ys, color=color)
        ax.fill_between(xs, ys, alpha=0.3, color=color)
    except np.linalg.LinAlgError:
        # KDE bandwidth matrix went singular (e.g. all samples identical).
        ax.hist(arr, bins=histogram_bins, color=color)
    ax.set_title(plot_title)


def plot_variable_to_base64(
    samples: torch.Tensor,
    variable: AnnotatedVariable,
    config: dict,
    title: Optional[str] = None,
    color: Optional[str] = None,
) -> str:
    """Render :func:`plot_variable` to an in-memory PNG and base64-encode it.

    Returns the *raw* base64 string with no ``data:image/png;base64,`` prefix.
    The prefix is the consumer's responsibility (added at the call site that
    drops the string into an ``<img src=...>`` tag).
    """
    viz_cfg = config.get("visualization", {})
    figure_size = tuple(viz_cfg.get("figure_size", (4, 3)))
    kde_points = viz_cfg.get("kde_points", 200)
    histogram_bins = viz_cfg.get("histogram_bins", 40)
    plot_color = color if color is not None else TAG_COLORS.get(variable.tag, "#888888")

    fig, ax = plt.subplots(figsize=figure_size)
    try:
        plot_variable(
            samples,
            variable,
            ax=ax,
            color=plot_color,
            kde_points=kde_points,
            histogram_bins=histogram_bins,
            title=title,
        )
        buf = io.BytesIO()
        fig.tight_layout()
        fig.savefig(buf, format="png", dpi=100)
        buf.seek(0)
        encoded = base64.b64encode(buf.read()).decode("utf-8")
    finally:
        plt.close(fig)
    return encoded


def build_cytoscape_elements(
    ditto_graph: DittoGraph,
    inference_result: InferenceResult,
    config: dict,
) -> List[dict]:
    """Build the ``elements`` list for ``cyto.Cytoscape``.

    Each node carries up to two base64 PNGs inline:

    * ``data.image_prior`` — the prior plot (for ``prior`` and ``latent``).
    * ``data.image_posterior`` — the posterior plot (for ``latent`` and
      ``observed``).

    ``data.image`` is also kept for backward compatibility and points at the
    "primary" plot for the node (prior for ``prior``, posterior for
    ``observed``, prior for ``latent`` — the tooltip handler uses both
    ``image_prior``/``image_posterior`` directly).
    """
    elements: List[dict] = []

    prior_color = TAG_COLORS["prior"]
    posterior_color = TAG_COLORS["observed"]

    for name, variable in ditto_graph.variables.items():
        prior_b64 = ""
        posterior_b64 = ""

        prior_tensor = inference_result.prior_samples.get(name)
        posterior_tensor = inference_result.posterior_samples.get(name)

        # Fall back to the primary ``tag`` field when no explicit ``tags``
        # frozenset was supplied (older callers / tests construct
        # ``AnnotatedVariable`` without it).
        effective_tags = variable.tags if variable.tags else frozenset({variable.tag})
        # ``latent`` implies a prior plot too: latents get both the prior
        # (from sampling the declared distribution) and the posterior (from
        # SVI). ``prior`` of course gets a prior plot. ``observed`` does not.
        show_prior = bool(effective_tags & {"prior", "latent"})
        show_posterior = bool(effective_tags & {"latent", "observed"})

        if show_prior and prior_tensor is not None:
            prior_b64 = plot_variable_to_base64(
                prior_tensor, variable, config,
                title=f"{name} — Prior", color=prior_color,
            )
        if show_posterior and posterior_tensor is not None:
            posterior_b64 = plot_variable_to_base64(
                posterior_tensor, variable, config,
                title=f"{name} — Posterior" if "latent" in effective_tags
                      else f"{name} (posterior predictive)",
                color=posterior_color,
            )

        primary = prior_b64 if prior_b64 else posterior_b64

        elements.append(
            {
                "data": {
                    "id": name,
                    "label": name,
                    "tag": variable.tag,
                    "tags": sorted(effective_tags),
                    "raw_expression": variable.raw_expression,
                    "image": primary,
                    "image_prior": prior_b64,
                    "image_posterior": posterior_b64,
                }
            }
        )

    for u, v in ditto_graph.graph.edges():
        elements.append(
            {"data": {"id": f"{u}__{v}", "source": u, "target": v}}
        )

    return elements


def _build_stylesheet(config: dict) -> List[dict]:
    """Cytoscape stylesheet: one selector per tag color, plus edge styling."""
    viz_cfg = config.get("visualization", {})
    prior_color = viz_cfg.get("prior_color", TAG_COLORS["prior"])
    observed_color = viz_cfg.get("observed_color", TAG_COLORS["observed"])
    latent_color = viz_cfg.get("latent_color", TAG_COLORS["latent"])

    return [
        {
            "selector": "node",
            "style": {
                "label": "data(label)",
                "text-valign": "center",
                "text-halign": "center",
                "color": "white",
                "font-size": "14px",
                "width": 60,
                "height": 60,
            },
        },
        {"selector": 'node[tag = "prior"]', "style": {"background-color": prior_color}},
        {"selector": 'node[tag = "observed"]', "style": {"background-color": observed_color}},
        {"selector": 'node[tag = "latent"]', "style": {"background-color": latent_color}},
        {
            "selector": "edge",
            "style": {
                "curve-style": "bezier",
                "target-arrow-shape": "triangle",
                "line-color": "#999",
                "target-arrow-color": "#999",
                "width": 2,
            },
        },
    ]


# ---------------------------------------------------------------------------
# Phase 6 helpers
# ---------------------------------------------------------------------------


def _build_lookups(elements: List[dict]) -> Tuple[Dict[str, str], Dict[str, str], Dict[str, str]]:
    """Extract per-node base64 image and raw-expression lookup tables.

    Returns ``(prior_lookup, posterior_lookup, expression_lookup)``.
    """
    prior_lookup: Dict[str, str] = {}
    posterior_lookup: Dict[str, str] = {}
    expression_lookup: Dict[str, str] = {}
    for elem in elements:
        if "source" in elem["data"]:
            continue
        node_id = elem["data"]["id"]
        prior_lookup[node_id] = elem["data"].get("image_prior", "")
        posterior_lookup[node_id] = elem["data"].get("image_posterior", "")
        expression_lookup[node_id] = elem["data"].get("raw_expression", "")
    return prior_lookup, posterior_lookup, expression_lookup


def _write_temp_py_file(decoded: bytes) -> str:
    """Write ``decoded`` bytes to a NamedTemporaryFile and return its path.

    ``delete=False`` so the file persists past the ``close()`` call (Dash
    needs to be able to read it again from a separate code path).
    """
    fh = tempfile.NamedTemporaryFile(
        mode="wb", suffix=".py", delete=False
    )
    try:
        fh.write(decoded)
    finally:
        fh.close()
    return fh.name


def _decode_upload_contents(contents: str) -> bytes:
    """Decode a ``dcc.Upload`` ``contents`` string into raw bytes.

    The format is ``data:<mime>;base64,<payload>``.
    """
    if "," not in contents:
        raise ValueError("upload contents missing base64 payload separator")
    _header, payload = contents.split(",", 1)
    return base64.b64decode(payload)


def _load_data_file(filename: str, decoded: bytes) -> dict:
    """Parse an uploaded data file into a Pyro-compatible structure.

    Returns a dict with keys:

    * ``"args"``: a list of tensors to pass positionally.
    * ``"kwargs"``: a mapping of kwarg name to tensor.
    * ``"summary"``: a one-line human-readable description (for the UI).

    The conversion conventions are:

    * ``.csv``: read with ``pandas.read_csv``. If a column called ``obs`` or
      ``y`` is present, that column becomes ``kwargs["obs"]`` and the
      remaining columns are concatenated column-wise into ``args[0]``.
      Otherwise the whole frame becomes ``args[0]``.
    * ``.npy``: ``numpy.load`` then ``torch.from_numpy``; result is stored as
      ``args[0]``.
    * ``.pkl``: ``pickle.load``. The result must be either a tuple
      ``(args, kwargs)`` or a single tensor (in which case it becomes
      ``args[0]``).
    """
    lower = filename.lower()
    if lower.endswith(".csv"):
        import pandas as pd
        frame = pd.read_csv(io.BytesIO(decoded))
        obs_col = None
        for candidate in ("obs", "y", "Y"):
            if candidate in frame.columns:
                obs_col = candidate
                break
        if obs_col is not None:
            obs_tensor = torch.tensor(
                frame[obs_col].to_numpy(), dtype=torch.float32
            )
            feature_cols = [c for c in frame.columns if c != obs_col]
            args: list = []
            if feature_cols:
                if len(feature_cols) == 1:
                    args.append(torch.tensor(
                        frame[feature_cols[0]].to_numpy(), dtype=torch.float32
                    ))
                else:
                    args.append(torch.tensor(
                        frame[feature_cols].to_numpy(), dtype=torch.float32
                    ))
            kwargs = {"obs": obs_tensor}
            summary = (
                f"CSV {filename}: {frame.shape[0]} rows x {frame.shape[1]} cols; "
                f"obs<-'{obs_col}'"
            )
        else:
            args = [torch.tensor(frame.to_numpy(), dtype=torch.float32)]
            kwargs = {}
            summary = (
                f"CSV {filename}: {frame.shape[0]} rows x {frame.shape[1]} cols"
            )
        return {"args": args, "kwargs": kwargs, "summary": summary}

    if lower.endswith(".npy"):
        arr = np.load(io.BytesIO(decoded), allow_pickle=False)
        tensor = torch.from_numpy(arr)
        return {
            "args": [tensor],
            "kwargs": {},
            "summary": f"NPY {filename}: shape {tuple(tensor.shape)}",
        }

    if lower.endswith(".pkl"):
        obj = pickle.loads(decoded)
        if isinstance(obj, tuple) and len(obj) == 2 and isinstance(obj[1], dict):
            args = list(obj[0])
            kwargs = dict(obj[1])
            summary = (
                f"PKL {filename}: ({len(args)} positional, "
                f"{sorted(kwargs)} kwargs)"
            )
        elif isinstance(obj, torch.Tensor):
            args = [obj]
            kwargs = {}
            summary = f"PKL {filename}: tensor shape {tuple(obj.shape)}"
        else:
            raise ValueError(
                "PKL file must contain either (args, kwargs) tuple or a Tensor."
            )
        return {"args": args, "kwargs": kwargs, "summary": summary}

    raise ValueError(
        f"Unsupported data file extension for {filename!r} "
        f"(expected .csv, .npy, or .pkl)."
    )


def _serialize_state(
    ditto_graph: Optional[DittoGraph],
    inference_result: Optional[InferenceResult],
    elements: List[dict],
    filepath: Optional[str],
) -> dict:
    """Pack the in-memory model state into a JSON-serialisable dict.

    Tensors are converted to nested Python lists. Only the bits the callbacks
    need to function are stored; the heavy ``ditto_graph`` and ``user_module``
    objects live in module-level state via :class:`_RuntimeState`.
    """
    if ditto_graph is None or inference_result is None:
        return {
            "filepath": filepath,
            "variables": {},
            "elements": elements,
            "edges": [],
        }
    variables_payload = {}
    for name, var in ditto_graph.variables.items():
        variables_payload[name] = {
            "name": var.name,
            "tag": var.tag,
            "tags": sorted(var.tags),
            "raw_expression": var.raw_expression,
            "is_sample_call": var.is_sample_call,
        }
    return {
        "filepath": filepath,
        "variables": variables_payload,
        "elements": elements,
        "edges": [
            {"source": u, "target": v}
            for u, v in ditto_graph.graph.edges()
        ],
    }


class _RuntimeState:
    """Server-side mutable state shared across callbacks.

    ``dcc.Store`` is great for JSON-serialisable bits (the elements list,
    user-edited prior expressions, etc.) but it cannot hold live Python
    objects. The :class:`DittoGraph`, the ``user_module`` import, and the
    last :class:`InferenceResult` therefore live here in process memory
    keyed by app instance.
    """

    def __init__(self) -> None:
        self.ditto_graph: Optional[DittoGraph] = None
        self.user_module = None
        self.inference_result: Optional[InferenceResult] = None
        self.filepath: Optional[str] = None
        self.data_args: Optional[tuple] = None
        self.data_kwargs: Optional[dict] = None

    def has_model(self) -> bool:
        return self.ditto_graph is not None and self.inference_result is not None


def _file_dialog_pick() -> Optional[str]:
    """Open a native file dialog and return the chosen path (or ``None``).

    Runs ``tkinter.filedialog.askopenfilename`` in a freshly created hidden
    Tk root. Returns ``None`` if the user cancelled. Raises :class:`OSError`
    or :class:`ImportError` if Tk is not available (e.g. on a headless
    server).
    """
    import tkinter
    from tkinter import filedialog

    root = tkinter.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    try:
        path = filedialog.askopenfilename(
            title="Open Pyro source file",
            filetypes=[("Python files", "*.py"), ("All files", "*.*")],
        )
    finally:
        root.destroy()
    return path or None


def _run_pipeline(
    filepath: str,
    config: dict,
    runtime: _RuntimeState,
    data_override: Optional[Tuple[tuple, dict]] = None,
) -> Tuple[DittoGraph, InferenceResult, List[dict]]:
    """Run the full parse → build → load → inference pipeline.

    The result is also stashed on ``runtime`` so subsequent edit/refresh
    callbacks can find it. ``data_override``, if supplied, overrides the
    user's ``get_data()`` for the duration of inference by monkey-patching
    the loaded module.
    """
    # Local imports avoid pulling Pyro into module import time when the only
    # caller is e.g. a lightweight test of the plotting helpers.
    from ditto.parser import parse_file
    from ditto.graph_builder import build_graph
    from ditto.inference_engine import load_user_module, run_inference

    variables = parse_file(filepath)
    ditto_graph = build_graph(variables, filepath=filepath)
    user_module = load_user_module(filepath)

    if data_override is not None:
        args, kwargs = data_override
        # Replace get_data so the inference engine picks up the override.
        user_module.get_data = lambda args=args, kwargs=kwargs: (args, kwargs)

    inference_result = run_inference(ditto_graph, user_module, config)
    elements = build_cytoscape_elements(ditto_graph, inference_result, config)

    runtime.ditto_graph = ditto_graph
    runtime.user_module = user_module
    runtime.inference_result = inference_result
    runtime.filepath = filepath
    if data_override is not None:
        runtime.data_args, runtime.data_kwargs = data_override
    return ditto_graph, inference_result, elements


# ---------------------------------------------------------------------------
# Source-file rewriting (Phase 6.4 — "Apply Prior" writes back to the .py)
# ---------------------------------------------------------------------------


def _rewrite_prior_in_file(
    filepath: str, var: AnnotatedVariable, new_expression: str
) -> None:
    """Overwrite the RHS of ``var``'s annotated assignment in ``filepath``.

    Finds the assignment node that follows the annotation comment for ``var``,
    replaces everything between ``lineno`` and ``end_lineno`` with a single new
    line, and writes the file back in place.  Multi-line assignments (e.g.
    parenthesised arguments) are collapsed to one line — which is fine because
    Ditto's parser round-trips through ``ast.unparse`` anyway.
    """
    with open(filepath, "r", encoding="utf-8") as fh:
        lines = fh.readlines()
    source = "".join(lines)
    tree = ast.parse(source, filename=filepath)

    annotation_line = var.line_number  # 1-indexed

    target_node = None
    for node in ast.walk(tree):
        if not hasattr(node, "lineno") or node.lineno <= annotation_line:
            continue
        if isinstance(node, ast.Assign):
            if (
                len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name)
                and node.targets[0].id == var.name
            ):
                if target_node is None or node.lineno < target_node.lineno:
                    target_node = node
        elif isinstance(node, ast.AnnAssign):
            if isinstance(node.target, ast.Name) and node.target.id == var.name:
                if target_node is None or node.lineno < target_node.lineno:
                    target_node = node

    if target_node is None:
        raise ValueError(
            f"Cannot locate the assignment for '{var.name}' after annotation "
            f"at line {annotation_line} in {filepath!r}."
        )

    # Preserve the original indentation from the first line of the assignment.
    start_idx = target_node.lineno - 1  # convert to 0-indexed
    indent = lines[start_idx][: len(lines[start_idx]) - len(lines[start_idx].lstrip())]

    if isinstance(target_node, ast.AnnAssign) and target_node.annotation is not None:
        ann = ast.unparse(target_node.annotation)
        new_line = f"{indent}{var.name}: {ann} = {new_expression}\n"
    else:
        new_line = f"{indent}{var.name} = {new_expression}\n"

    # Replace the (possibly multi-line) original with the new single line.
    end_idx = target_node.end_lineno  # 1-indexed inclusive → 0-indexed exclusive
    lines[start_idx:end_idx] = [new_line]

    with open(filepath, "w", encoding="utf-8") as fh:
        fh.writelines(lines)


# ---------------------------------------------------------------------------
# Layout / app construction
# ---------------------------------------------------------------------------


def _empty_canvas_message() -> html.Div:
    return html.Div(
        "Drop a .py file here or use the 'Open File…' button to begin.",
        style={
            "color": "#888",
            "fontSize": "18px",
            "textAlign": "center",
            "padding": "32px",
        },
    )


def _initial_side_panel(message: str = "Hover or click a node to see its distribution.") -> html.Div:
    return html.Div(message)


def _hidden_edit_stubs() -> html.Div:
    """Hidden placeholder components for prior-edit controls.

    Dash callbacks targeting ``prior-edit-textarea``/``apply-prior-btn``/
    ``prior-edit-msg`` need those IDs to exist in the layout at firing time,
    even when the currently selected variable isn't tagged ``prior``. We
    render a ``display: none`` stub so the components remain in the tree
    and callbacks register cleanly.
    """
    return html.Div(
        style={"display": "none"},
        children=[
            dcc.Textarea(id="prior-edit-textarea", value=""),
            html.Button("Apply Prior", id="apply-prior-btn", n_clicks=0),
            html.Span(id="prior-edit-msg"),
        ],
    )


def _hidden_data_stubs() -> html.Div:
    """Hidden placeholder components for the data-upload affordance.

    Mirrors :func:`_hidden_edit_stubs` for ``upload-data``/``data-summary``
    so the upload callback can still register for any variable selection
    even though the visible affordance is only rendered for ``observed``
    variables.
    """
    return html.Div(
        style={"display": "none"},
        children=[
            dcc.Upload(id="upload-data", children=html.Div("")),
            html.Div(id="data-summary", children=""),
        ],
    )


def _toolbar(file_loaded: bool) -> html.Div:
    return html.Div(
        style={
            "display": "flex",
            "alignItems": "center",
            "gap": "8px",
            "padding": "8px",
            "borderBottom": "1px solid #eee",
            "background": "#fafafa",
        },
        children=[
            html.H2(
                "Ditto: Pyro DAG Visualizer",
                style={"margin": 0, "flex": "1", "fontSize": "20px"},
            ),
            html.Button(
                "Open File…",
                id="open-file-btn",
                n_clicks=0,
                style={"padding": "6px 12px"},
            ),
            html.Button(
                "Refresh",
                id="refresh-btn",
                n_clicks=0,
                disabled=not file_loaded,
                style={"padding": "6px 12px"},
            ),
            html.Span(
                id="status-msg",
                style={"marginLeft": "12px", "color": "#555", "fontSize": "13px"},
            ),
        ],
    )


def create_dash_app(
    ditto_graph: Optional[DittoGraph],
    inference_result: Optional[InferenceResult],
    config: dict,
    filepath: Optional[str] = None,
) -> dash.Dash:
    """Construct the Dash application (does not start the server).

    When ``ditto_graph`` and ``inference_result`` are ``None`` the app boots
    in "empty" mode (Phase 6.1). The user can then load a file at runtime
    via drag-and-drop or the system file dialog, edit priors, load data,
    and refresh inference — all without restarting the server.
    """
    runtime = _RuntimeState()

    if ditto_graph is not None and inference_result is not None:
        elements = build_cytoscape_elements(ditto_graph, inference_result, config)
        runtime.ditto_graph = ditto_graph
        runtime.inference_result = inference_result
        runtime.filepath = filepath
    else:
        elements = []

    stylesheet = _build_stylesheet(config)

    file_loaded = runtime.has_model()
    initial_state = _serialize_state(
        runtime.ditto_graph, runtime.inference_result, elements, runtime.filepath
    )

    app = dash.Dash(__name__, suppress_callback_exceptions=True)
    app.title = "Ditto"

    upload_overlay_style = {
        "border": "2px dashed #aaa",
        "borderRadius": "8px",
        "textAlign": "center",
        "padding": "24px",
        "margin": "12px",
        "color": "#666",
        "background": "rgba(255,255,255,0.6)",
    }
    upload_overlay_style_loaded = {
        **upload_overlay_style,
        "padding": "6px",
        "fontSize": "12px",
        "borderColor": "#ccc",
        "color": "#888",
    }

    # ------------------------------------------------------------------
    # Stores — single source of truth for runtime mutable state.
    # ------------------------------------------------------------------
    stores = [
        # The serialized model state (variables + raw_expression dict +
        # elements list). Updated by upload, refresh, and prior-edit
        # callbacks; read by the cytoscape renderer.
        dcc.Store(id="model-store", storage_type="memory", data=initial_state),
        # The path chosen by the system file dialog (server-side callback).
        dcc.Store(id="file-dialog-store", storage_type="memory"),
        # Current selected node (for prior edit panel).
        dcc.Store(id="selected-node-store", storage_type="memory"),
        # Currently loaded data (args + kwargs serialised to lists).
        dcc.Store(id="data-store", storage_type="memory"),
        # Edited prior expressions (variable_name -> raw_expression).
        dcc.Store(id="prior-edit-store", storage_type="memory", data={}),
        # Trigger pulses to coordinate refresh with the inference run.
        dcc.Store(id="refresh-trigger", storage_type="memory", data=0),
    ]

    # ------------------------------------------------------------------
    # Layout
    # ------------------------------------------------------------------
    app.layout = html.Div(
        style={"display": "flex", "flexDirection": "column", "height": "100vh"},
        children=[
            *stores,
            _toolbar(file_loaded),
            html.Div(
                style={"display": "flex", "flexDirection": "row", "flex": "1",
                        "minHeight": 0},
                children=[
                    # Left: DAG panel with drag-drop overlay.
                    html.Div(
                        style={"flex": "3", "borderRight": "1px solid #ddd",
                                "position": "relative", "display": "flex",
                                "flexDirection": "column", "minHeight": 0},
                        children=[
                            dcc.Upload(
                                id="upload-py",
                                children=html.Div([
                                    "Drop a .py file here or ",
                                    html.A("click to browse", style={"color": "#4C72B0"}),
                                ]),
                                style=(upload_overlay_style_loaded
                                        if file_loaded else upload_overlay_style),
                                multiple=False,
                                accept=".py",
                            ),
                            dcc.Loading(
                                id="dag-loading",
                                type="circle",
                                style={"flex": "1", "minHeight": 0},
                                children=html.Div(
                                    id="dag-container",
                                    style={"flex": "1", "minHeight": 0,
                                            "position": "relative"},
                                    children=[
                                        cyto.Cytoscape(
                                            id="cytoscape",
                                            layout={"name": "dagre"},
                                            autoRefreshLayout=True,
                                            userPanningEnabled=True,
                                            userZoomingEnabled=True,
                                            style={"width": "100%", "height": "100%",
                                                    "minHeight": "400px",
                                                    "display": "block" if file_loaded
                                                    else "none"},
                                            elements=elements,
                                            stylesheet=stylesheet,
                                        ),
                                        html.Div(
                                            id="empty-placeholder",
                                            children=(
                                                _empty_canvas_message()
                                                if not file_loaded else ""
                                            ),
                                            style={"display": "block" if not file_loaded
                                                   else "none"},
                                        ),
                                    ],
                                ),
                            ),
                        ],
                    ),
                    # Right: side panel with hover/click + per-variable edit
                    # / data upload (rendered conditionally based on tags).
                    html.Div(
                        id="side-panel",
                        style={"flex": "1", "padding": "12px",
                                "overflowY": "auto", "minWidth": "320px"},
                        children=[
                            html.Div(id="tooltip-div",
                                        children=_initial_side_panel()),
                        ],
                    ),
                ],
            ),
        ],
    )

    # ------------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------------

    @app.callback(
        Output("file-dialog-store", "data"),
        Input("open-file-btn", "n_clicks"),
        prevent_initial_call=True,
    )
    def _open_file_dialog(n_clicks):
        if not n_clicks:
            raise PreventUpdate
        try:
            path = _file_dialog_pick()
        except (ImportError, OSError) as exc:
            return {"error": f"File dialog unavailable: {exc}"}
        except Exception as exc:  # noqa: BLE001
            return {"error": f"File dialog error: {exc}"}
        if path is None:
            raise PreventUpdate
        return {"path": path}

    @app.callback(
        Output("model-store", "data"),
        Output("cytoscape", "elements"),
        Output("status-msg", "children"),
        Output("upload-py", "style"),
        Output("cytoscape", "style"),
        Output("empty-placeholder", "style"),
        Output("refresh-btn", "disabled"),
        Output("prior-edit-store", "data"),
        Input("upload-py", "contents"),
        Input("file-dialog-store", "data"),
        Input("refresh-trigger", "data"),
        State("upload-py", "filename"),
        State("model-store", "data"),
        State("data-store", "data"),
        State("prior-edit-store", "data"),
        prevent_initial_call=True,
    )
    def _load_or_refresh(upload_contents, dialog_data, refresh_trigger,
                            upload_filename, current_state, data_state,
                            prior_edits):
        ctx = dash.callback_context
        if not ctx.triggered:
            raise PreventUpdate
        trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]

        # 1. Determine the source filepath for this run.
        filepath: Optional[str] = None
        new_load = False
        if trigger_id == "upload-py" and upload_contents:
            try:
                decoded = _decode_upload_contents(upload_contents)
                filepath = _write_temp_py_file(decoded)
            except Exception as exc:  # noqa: BLE001
                return (
                    no_update, no_update,
                    f"Upload failed: {exc}",
                    upload_overlay_style,
                    no_update, no_update, no_update, no_update,
                )
            new_load = True
        elif trigger_id == "file-dialog-store" and dialog_data:
            if "error" in dialog_data:
                return (
                    no_update, no_update,
                    dialog_data["error"],
                    no_update, no_update, no_update, no_update, no_update,
                )
            filepath = dialog_data.get("path")
            if not filepath:
                raise PreventUpdate
            new_load = True
        elif trigger_id == "refresh-trigger":
            if not runtime.filepath:
                return (
                    no_update, no_update,
                    "Refresh requires a loaded file.",
                    no_update, no_update, no_update, no_update, no_update,
                )
            filepath = runtime.filepath
            new_load = False
            # Flush any pending textarea edits to the source file BEFORE we
            # re-parse and re-run inference. Without this, a user could type
            # a new prior expression, click Refresh, and see no change —
            # because Refresh by itself doesn't write to disk.
            pending = prior_edits or {}
            if pending and runtime.ditto_graph is not None:
                bad = []
                for var_name, expression in list(pending.items()):
                    var = runtime.ditto_graph.variables.get(var_name)
                    if var is None:
                        continue
                    expr = (expression or "").strip()
                    if not expr:
                        continue
                    try:
                        ast.parse(expr, mode="eval")
                    except SyntaxError as exc:
                        bad.append(f"{var_name}: {exc.msg}")
                        continue
                    try:
                        _rewrite_prior_in_file(filepath, var, expr)
                    except Exception as exc:  # noqa: BLE001
                        bad.append(f"{var_name}: {exc}")
                if bad:
                    return (
                        no_update, no_update,
                        "Refresh failed to apply edits: " + "; ".join(bad),
                        no_update, no_update, no_update, no_update, no_update,
                    )
        else:
            raise PreventUpdate

        # 2. Determine the data override (from data-store, if present).
        data_override = None
        if data_state and data_state.get("args") is not None:
            try:
                args = tuple(
                    torch.tensor(a) if not isinstance(a, torch.Tensor) else a
                    for a in data_state["args"]
                )
                kwargs = {
                    k: (torch.tensor(v) if not isinstance(v, torch.Tensor) else v)
                    for k, v in (data_state.get("kwargs") or {}).items()
                }
                data_override = (args, kwargs)
            except Exception as exc:  # noqa: BLE001
                return (
                    no_update, no_update,
                    f"Data conversion failed: {exc}",
                    no_update, no_update, no_update, no_update, no_update,
                )

        # 3. Run the pipeline from the (possibly edited) file.
        try:
            ditto_graph, inference_result, elements_local = _run_pipeline(
                filepath, config, runtime, data_override=data_override,
            )
        except Exception as exc:  # noqa: BLE001
            traceback.print_exc()
            return (
                no_update, no_update,
                f"Inference failed: {exc}",
                no_update, no_update, no_update, no_update, no_update,
            )

        new_state = _serialize_state(
            runtime.ditto_graph, runtime.inference_result,
            elements_local, runtime.filepath,
        )

        # 4. After a brand-new load OR a refresh (which has now flushed any
        # pending edits to disk), drop the prior-edit-store entries. The
        # newly re-parsed elements list is now the source of truth for the
        # textarea's rendered value.
        next_prior_edits = {}

        cytoscape_style = {
            "width": "100%", "height": "100%", "minHeight": "400px",
            "display": "block",
        }
        empty_style = {"display": "none"}

        return (
            new_state,
            elements_local,
            f"Loaded {os.path.basename(runtime.filepath)} "
            f"({len(runtime.ditto_graph.variables)} variables)",
            upload_overlay_style_loaded,
            cytoscape_style,
            empty_style,
            False,
            next_prior_edits,
        )

    @app.callback(
        Output("refresh-trigger", "data"),
        Input("refresh-btn", "n_clicks"),
        State("refresh-trigger", "data"),
        prevent_initial_call=True,
    )
    def _bump_refresh(n_clicks, current):
        if not n_clicks:
            raise PreventUpdate
        return (current or 0) + 1

    @app.callback(
        Output("prior-edit-store", "data", allow_duplicate=True),
        Input("prior-edit-textarea", "value"),
        State("selected-node-store", "data"),
        State("model-store", "data"),
        State("prior-edit-store", "data"),
        prevent_initial_call=True,
    )
    def _capture_prior_edit(value, selected, model_state, prior_edits):
        """Persist user textarea edits into ``prior-edit-store``.

        Without this, typing into the textarea is purely visual: a Refresh
        click would re-read the file's original expression and discard the
        user's intended edit. Storing the in-flight value here lets Refresh
        re-apply it.
        """
        if not selected or not selected.get("id"):
            raise PreventUpdate
        if not model_state or "variables" not in model_state:
            raise PreventUpdate
        node_id = selected["id"]
        if node_id not in model_state["variables"]:
            raise PreventUpdate
        # Only track edits for nodes that actually expose the editor —
        # i.e. nodes tagged ``prior``. Other nodes should never touch
        # the store (their textarea is the hidden stub with value="").
        var_meta = model_state["variables"][node_id]
        tags = set(var_meta.get("tags", [var_meta.get("tag", "")]))
        if "prior" not in tags:
            raise PreventUpdate

        edits = dict(prior_edits or {})
        # Pull the canonical "on-disk" expression from the model store so we
        # can drop entries that match it (no point storing a no-op edit).
        elements_local = model_state.get("elements", [])
        _, _, expression_lookup = _build_lookups(elements_local)
        canonical = expression_lookup.get(node_id, "")
        new_value = (value or "").rstrip("\n")

        if new_value == canonical or new_value == "":
            edits.pop(node_id, None)
        else:
            edits[node_id] = new_value
        return edits

    @app.callback(
        Output("data-store", "data"),
        Output("data-summary", "children"),
        Input("upload-data", "contents"),
        State("upload-data", "filename"),
        prevent_initial_call=True,
    )
    def _load_data_upload(contents, filename):
        if not contents or not filename:
            raise PreventUpdate
        try:
            decoded = _decode_upload_contents(contents)
            parsed = _load_data_file(filename, decoded)
        except Exception as exc:  # noqa: BLE001
            return no_update, html.Span(
                f"Data load failed: {exc}", style={"color": "#c0392b"}
            )

        # Tensors are not JSON-serialisable; convert to nested lists for
        # the dcc.Store. Reconstructed in the inference callback.
        try:
            args_serialised = [
                t.detach().cpu().tolist() if isinstance(t, torch.Tensor) else t
                for t in parsed["args"]
            ]
            kwargs_serialised = {
                k: (v.detach().cpu().tolist() if isinstance(v, torch.Tensor) else v)
                for k, v in parsed["kwargs"].items()
            }
        except Exception as exc:  # noqa: BLE001
            return no_update, html.Span(
                f"Data serialisation failed: {exc}", style={"color": "#c0392b"}
            )

        return (
            {"args": args_serialised, "kwargs": kwargs_serialised,
             "summary": parsed["summary"]},
            html.Span(parsed["summary"] + " (click Refresh to use)",
                      style={"color": "#2c3e50"}),
        )

    @app.callback(
        Output("selected-node-store", "data"),
        Input("cytoscape", "tapNodeData"),
        prevent_initial_call=True,
    )
    def _store_selected_node(node_data):
        if not node_data:
            raise PreventUpdate
        return {
            "id": node_data.get("id"),
            "tag": node_data.get("tag"),
            "tags": node_data.get("tags", []),
        }

    @app.callback(
        Output("tooltip-div", "children"),
        Input("cytoscape", "mouseoverNodeData"),
        Input("selected-node-store", "data"),
        Input("model-store", "data"),
        Input("data-store", "data"),
        State("prior-edit-store", "data"),
        prevent_initial_call=False,
    )
    def _render_side_panel(hover_data, selected, model_state, data_state,
                            prior_edits):
        ctx = dash.callback_context
        trigger_id = ctx.triggered[0]["prop_id"].split(".")[0] if ctx.triggered else ""

        if not model_state or not model_state.get("variables"):
            return [
                _initial_side_panel(
                    "Load a .py file to see distributions and edit priors."
                ),
                _hidden_edit_stubs(),
                _hidden_data_stubs(),
            ]

        elements_local = model_state.get("elements", [])
        prior_lookup, posterior_lookup, expression_lookup = _build_lookups(
            elements_local
        )
        variables_payload = model_state.get("variables", {})

        # Selected (clicked) node takes priority over hover, because clicking
        # is the "stickier" interaction users use to edit.
        node_id = None
        if trigger_id == "cytoscape" and hover_data:
            node_id = hover_data.get("id")
        elif selected and selected.get("id"):
            node_id = selected.get("id")
        elif hover_data:
            node_id = hover_data.get("id")

        if not node_id or node_id not in variables_payload:
            return [
                _initial_side_panel(),
                _hidden_edit_stubs(),
                _hidden_data_stubs(),
            ]

        var_meta = variables_payload[node_id]
        tag = var_meta.get("tag", "")
        tags = set(var_meta.get("tags", [tag]))
        prior_b64 = prior_lookup.get(node_id, "")
        posterior_b64 = posterior_lookup.get(node_id, "")

        children: list = [html.H3(f"{node_id} ({tag})")]
        if prior_b64:
            if "latent" in tags or "observed" in tags:
                children.append(html.H4("Prior"))
            children.append(html.Img(
                src=f"data:image/png;base64,{prior_b64}",
                style={"maxWidth": "100%"},
            ))
        if posterior_b64:
            if "prior" in tags:
                children.append(html.H4("Posterior"))
            children.append(html.Img(
                src=f"data:image/png;base64,{posterior_b64}",
                style={"maxWidth": "100%"},
            ))
        if not prior_b64 and not posterior_b64:
            children.append(html.P("No samples available for this variable."))

        # Editable prior expression: only shown for nodes explicitly tagged
        # ``prior``. Pure ``latent`` / ``observed`` nodes don't get this
        # affordance — their priors are derived from the model's structure
        # rather than user-editable on the canvas.
        if "prior" in tags:
            # Prefer any pending (un-applied) edit over the persisted file
            # value so refresh keeps the user's typed text on screen.
            pending_edits = prior_edits or {}
            current_value = pending_edits.get(
                node_id, expression_lookup.get(node_id, "")
            )
            children.append(html.Hr())
            children.append(html.H4("Edit prior"))
            children.append(dcc.Textarea(
                id="prior-edit-textarea",
                value=current_value,
                style={"width": "100%", "height": "70px",
                        "fontFamily": "monospace", "fontSize": "12px"},
            ))
            children.append(html.Div(
                style={"display": "flex", "gap": "8px",
                        "alignItems": "center", "marginTop": "6px"},
                children=[
                    html.Button("Apply Prior", id="apply-prior-btn",
                                 n_clicks=0, style={"padding": "4px 10px"}),
                    html.Span(id="prior-edit-msg",
                                 style={"fontSize": "12px", "color": "#555"}),
                ],
            ))
        else:
            children.append(_hidden_edit_stubs())

        # Data upload: only shown for nodes explicitly tagged ``observed``.
        if "observed" in tags:
            children.append(html.Hr())
            children.append(html.H4("Data"))
            children.append(dcc.Upload(
                id="upload-data",
                children=html.Div([
                    "Drop a data file (.csv, .npy, .pkl) or ",
                    html.A("click to browse", style={"color": "#4C72B0"}),
                ]),
                style={
                    "border": "1px dashed #aaa",
                    "borderRadius": "6px",
                    "padding": "10px",
                    "textAlign": "center",
                    "color": "#666",
                    "fontSize": "13px",
                },
                multiple=False,
                accept=".csv,.npy,.pkl",
            ))
            data_summary_msg = (
                (data_state or {}).get("summary")
                or "No data file loaded; using model's get_data()."
            )
            if data_state and data_state.get("summary"):
                data_summary_msg = (
                    f"{data_state['summary']} (click Refresh to use)"
                )
            children.append(html.Div(
                id="data-summary",
                style={"fontSize": "12px", "color": "#444",
                        "marginTop": "6px"},
                children=data_summary_msg,
            ))
        else:
            children.append(_hidden_data_stubs())

        return html.Div(children)

    @app.callback(
        Output("prior-edit-msg", "children"),
        Output("prior-edit-store", "data", allow_duplicate=True),
        Output("model-store", "data", allow_duplicate=True),
        Output("cytoscape", "elements", allow_duplicate=True),
        Input("apply-prior-btn", "n_clicks"),
        State("prior-edit-textarea", "value"),
        State("selected-node-store", "data"),
        State("prior-edit-store", "data"),
        prevent_initial_call=True,
    )
    def _apply_prior(n_clicks, new_expression, selected, edits_state):
        if not n_clicks or not selected or not selected.get("id"):
            raise PreventUpdate
        node_id = selected["id"]
        new_expression = (new_expression or "").strip()

        if not runtime.has_model() or runtime.ditto_graph is None:
            return (
                html.Span("No model loaded.", style={"color": "#c0392b"}),
                no_update, no_update, no_update,
            )
        if node_id not in runtime.ditto_graph.variables:
            return (
                html.Span(f"Unknown variable {node_id!r}.",
                          style={"color": "#c0392b"}),
                no_update, no_update, no_update,
            )
        if not runtime.filepath:
            return (
                html.Span("Cannot apply: no source file path is known.",
                          style={"color": "#c0392b"}),
                no_update, no_update, no_update,
            )

        # Validate the expression before touching the file.
        try:
            ast.parse(new_expression, mode="eval")
        except SyntaxError as exc:
            return (
                html.Span(f"Syntax error: {exc.msg}", style={"color": "#c0392b"}),
                no_update, no_update, no_update,
            )

        # Write the new expression into the source file so the change persists
        # and is picked up by the re-parse below.
        var = runtime.ditto_graph.variables[node_id]
        try:
            _rewrite_prior_in_file(runtime.filepath, var, new_expression)
        except Exception as exc:  # noqa: BLE001
            return (
                html.Span(f"File write failed: {exc}", style={"color": "#c0392b"}),
                no_update, no_update, no_update,
            )

        # Re-run the full pipeline (parse → build → inference) so that both
        # the prior plot AND the posterior reflect the new distribution.
        try:
            _, _, elements_local = _run_pipeline(runtime.filepath, config, runtime)
        except Exception as exc:  # noqa: BLE001
            traceback.print_exc()
            return (
                html.Span(f"Inference failed: {exc}", style={"color": "#c0392b"}),
                no_update, no_update, no_update,
            )

        new_state = _serialize_state(
            runtime.ditto_graph, runtime.inference_result,
            elements_local, runtime.filepath,
        )
        return (
            html.Span(
                "Prior updated — prior and posterior recomputed.",
                style={"color": "#27ae60"},
            ),
            {},  # clear prior-edit-store; the change now lives in the file
            new_state,
            elements_local,
        )

    return app


def _re_sample_prior(runtime: _RuntimeState, node_id: str) -> None:
    """Re-sample the prior for ``node_id`` after its expression was edited."""
    from ditto.inference_engine import sample_prior

    if runtime.ditto_graph is None or runtime.inference_result is None:
        return
    var = runtime.ditto_graph.variables[node_id]
    if "prior" in var.tags or var.tag in ("prior", "latent"):
        new_samples = sample_prior(var, runtime.inference_result.num_samples)
        runtime.inference_result.prior_samples[node_id] = new_samples


def render_dag_static(
    ditto_graph: DittoGraph,
    inference_result: InferenceResult,
    config: dict,
    output_path: str,
) -> None:
    """Optional static export of the DAG to a single PNG.

    Uses ``networkx`` for layout and embeds a tiny KDE/histogram next to each
    node. Useful for headless reports where the Dash app isn't appropriate.
    """
    viz_cfg = config.get("visualization", {})
    figure_size = tuple(viz_cfg.get("figure_size", (12, 8)))
    dpi = config.get("export", {}).get("dpi", 150)

    fig, ax = plt.subplots(figsize=figure_size)
    pos = nx.spring_layout(ditto_graph.graph, seed=0)
    node_colors = [
        TAG_COLORS.get(ditto_graph.variables[n].tag, "#888888")
        for n in ditto_graph.graph.nodes()
    ]
    nx.draw(
        ditto_graph.graph,
        pos=pos,
        ax=ax,
        with_labels=True,
        node_color=node_colors,
        node_size=1500,
        font_color="white",
        arrows=True,
    )
    ax.set_title("Ditto DAG")
    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)
