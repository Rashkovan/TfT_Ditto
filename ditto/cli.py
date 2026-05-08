"""Command-line entry point for Ditto.

Wires together parsing, graph construction, inference and the Dash server.
Designed to be installed as a console script via ``pyproject.toml``.
"""
from __future__ import annotations

import argparse
import sys
from typing import Optional

import yaml

#: Built-in defaults used when no config file is present (or some keys are
#: missing). Matches the structure documented in ``ditto.yaml``.
DEFAULT_CONFIG: dict = {
    "inference": {
        "svi_steps": 2000,
        "learning_rate": 0.01,
        "num_samples": 1000,
    },
    "visualization": {
        "kde_points": 200,
        "histogram_bins": 40,
        "figure_size": [12, 8],
        "prior_color": "#4C72B0",
        "observed_color": "#DD8452",
        "latent_color": "#55A868",
    },
    "server": {
        "port": 8050,
        "debug": False,
    },
    "export": {
        "path": None,
        "format": "png",
        "dpi": 150,
    },
}


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge ``override`` into a copy of ``base``.

    Used to layer the user's YAML on top of :data:`DEFAULT_CONFIG` so a
    partial config doesn't leave unrelated keys missing.
    """
    out = dict(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def load_config(config_path: str) -> dict:
    """Load ``config_path`` and merge it with :data:`DEFAULT_CONFIG`.

    Missing files are tolerated (returns a copy of the defaults), so first-
    time users can run ``ditto`` without authoring a YAML file.
    """
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            user_cfg = yaml.safe_load(f) or {}
    except FileNotFoundError:
        return dict(DEFAULT_CONFIG)
    return _deep_merge(DEFAULT_CONFIG, user_cfg)


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ditto",
        description="Visualize the DAG of a Pyro probabilistic model.",
    )
    parser.add_argument(
        "filepath",
        nargs="?",
        default=None,
        help=(
            "Path to the user's annotated Pyro source file. Optional: if "
            "omitted, the Dash app launches with an empty canvas and the user "
            "can drag-and-drop or browse to a file."
        ),
    )
    parser.add_argument(
        "--config",
        default="ditto.yaml",
        help="Path to the YAML config file (default: ditto.yaml).",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=None,
        help="Port for the Dash server (overrides config; default 8050 if neither set).",
    )
    parser.add_argument(
        "--export",
        default=None,
        help="If set, write a static PNG of the DAG to this path and exit.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Parse + build graph + run inference, but don't start the server.",
    )
    return parser


def main(argv: Optional[list] = None) -> int:
    """Console entry point. Returns the process exit code."""
    parser = _build_arg_parser()
    args = parser.parse_args(argv)

    try:
        # Local imports keep ``import ditto`` cheap and let us produce a
        # cleaner error if a heavy dependency (torch, dash) is missing.
        from ditto.parser import parse_file
        from ditto.graph_builder import build_graph
        from ditto.inference_engine import load_user_module, run_inference
        from ditto.visualizer import create_dash_app, render_dag_static

        config = load_config(args.config)

        # Phase 6.1: filepath is optional. When omitted we launch the Dash
        # app in an "empty" state and let the user load a file at runtime
        # via drag-and-drop (6.2) or the system file dialog (6.3).
        if args.filepath is None:
            if args.export or args.dry_run:
                print(
                    "[ditto] error: --export and --dry-run require a filepath.",
                    file=sys.stderr,
                )
                return 1
            app = create_dash_app(None, None, config)
            port = args.port if args.port is not None else config["server"]["port"]
            app.run(debug=False, port=port)
            return 0

        variables = parse_file(args.filepath)
        ditto_graph = build_graph(variables, filepath=args.filepath)
        user_module = load_user_module(args.filepath)
        inference_result = run_inference(ditto_graph, user_module, config)

        if args.export:
            render_dag_static(ditto_graph, inference_result, config, args.export)
            print(f"[ditto] wrote static DAG to {args.export}")

        if args.dry_run:
            print(
                f"[ditto] dry run complete: "
                f"{len(ditto_graph.variables)} variables, "
                f"{ditto_graph.graph.number_of_edges()} edges."
            )
            return 0

        app = create_dash_app(ditto_graph, inference_result, config,
                              filepath=args.filepath)
        port = args.port if args.port is not None else config["server"]["port"]
        # ``app.run`` is the preferred call in Dash >= 2.11; ``app.run_server``
        # still works as an alias but emits a DeprecationWarning. We use
        # ``app.run`` to be future-proof.
        app.run(debug=False, port=port)
        return 0

    except Exception as exc:  # noqa: BLE001
        import traceback
        print(f"[ditto] error: {exc}", file=sys.stderr)
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
