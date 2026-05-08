"""Ditto: a DAG visualization tool for Pyro probabilistic models.

Top-level package. Exposes a programmatic ``run`` entry point so users can
launch the visualization from Python without going through the CLI.
"""
from typing import Optional

from ditto.models import AnnotatedVariable, DittoGraph, InferenceResult

__all__ = [
    "AnnotatedVariable",
    "DittoGraph",
    "InferenceResult",
    "run",
]

__version__ = "0.1.0"


def run(filepath: str, config: Optional[dict] = None, port: int = 8050) -> None:
    """Programmatic entry point for Ditto.

    Parses ``filepath``, builds the DAG, runs inference (sampling + optional
    SVI), and launches the Dash web application.

    Parameters
    ----------
    filepath:
        Path to the user's Pyro source file containing ``# !Ditto:`` annotations.
    config:
        Optional already-loaded configuration dictionary. If ``None`` defaults
        from :mod:`ditto.cli` are used.
    port:
        Port for the Dash development server.
    """
    # Imports are local so importing the package is cheap and avoids hard
    # dependencies (matplotlib, dash, etc.) for callers that only need the
    # dataclasses.
    from ditto.cli import DEFAULT_CONFIG
    from ditto.parser import parse_file
    from ditto.graph_builder import build_graph
    from ditto.inference_engine import load_user_module, run_inference
    from ditto.visualizer import create_dash_app

    cfg = config if config is not None else DEFAULT_CONFIG

    variables = parse_file(filepath)
    ditto_graph = build_graph(variables, filepath=filepath)
    user_module = load_user_module(filepath)
    inference_result = run_inference(ditto_graph, user_module, cfg)
    app = create_dash_app(ditto_graph, inference_result, cfg)
    app.run(debug=False, port=port)
