"""Tests for :mod:`ditto.visualizer`."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402

from ditto.graph_builder import build_graph
from ditto.models import AnnotatedVariable, InferenceResult
from ditto.visualizer import (
    build_cytoscape_elements,
    plot_variable,
    plot_variable_to_base64,
)


def _make_var(name="mu", tag="prior") -> AnnotatedVariable:
    return AnnotatedVariable(
        name=name, tag=tag, line_number=1, raw_expression="dist.Normal(0., 1.)"
    )


CONFIG = {
    "visualization": {
        "kde_points": 50,
        "histogram_bins": 20,
        "figure_size": [4, 3],
    }
}


def test_plot_variable_runs_without_error():
    samples = torch.randn(200)
    var = _make_var()
    fig, ax = plt.subplots()
    try:
        plot_variable(samples, var, ax=ax, color="#4C72B0", kde_points=50)
    finally:
        plt.close(fig)


def test_plot_variable_handles_zero_variance():
    # All samples identical -> KDE bandwidth becomes singular. Should fall
    # back to a histogram instead of raising.
    samples = torch.zeros(50)
    var = _make_var()
    fig, ax = plt.subplots()
    try:
        plot_variable(samples, var, ax=ax, color="#4C72B0", kde_points=50)
    finally:
        plt.close(fig)


def test_plot_variable_to_base64_returns_non_empty_string():
    samples = torch.randn(100)
    var = _make_var()
    encoded = plot_variable_to_base64(samples, var, CONFIG)
    assert isinstance(encoded, str)
    assert len(encoded) > 0
    # Critical: no data: prefix is included; that's the caller's job.
    assert not encoded.startswith("data:")


def test_build_cytoscape_elements_has_correct_shape():
    variables = [
        AnnotatedVariable("a", "prior", 1, "dist.Normal(0., 1.)"),
        AnnotatedVariable("b", "observed", 4, "a + 1"),
    ]
    ditto_graph = build_graph(variables)
    prior = {"a": torch.randn(50)}
    posterior = {"b": torch.randn(50)}
    result = InferenceResult(
        prior_samples=prior,
        posterior_samples=posterior,
        losses=[],
        num_samples=50,
    )
    elements = build_cytoscape_elements(ditto_graph, result, CONFIG)

    nodes = [e for e in elements if "source" not in e["data"]]
    edges = [e for e in elements if "source" in e["data"]]

    assert len(nodes) == 2
    assert len(edges) == 1
    assert {n["data"]["id"] for n in nodes} == {"a", "b"}
    edge = edges[0]
    assert edge["data"]["source"] == "a"
    assert edge["data"]["target"] == "b"
    # Node data carries the encoded thumbnail and tag.
    by_id = {n["data"]["id"]: n["data"] for n in nodes}
    # Prior node has prior image but no posterior image.
    assert by_id["a"]["image_prior"]
    assert by_id["a"]["image_posterior"] == ""
    # Observed node has posterior image but no prior image.
    assert by_id["b"]["image_posterior"]
    assert by_id["b"]["image_prior"] == ""
    for n in nodes:
        assert "image" in n["data"]
        assert "label" in n["data"]
        assert "tag" in n["data"]


def test_build_cytoscape_elements_latent_node_gets_two_images():
    """A ``latent`` variable should pre-compute BOTH a prior and posterior
    base64 thumbnail so the hover tooltip can stack them vertically."""
    variables = [
        AnnotatedVariable("mu", "latent", 1, "dist.Normal(0., 1.)"),
        AnnotatedVariable("x", "observed", 4, "mu + 1"),
    ]
    ditto_graph = build_graph(variables)
    prior = {"mu": torch.randn(50)}
    posterior = {"mu": torch.randn(50), "x": torch.randn(50)}
    result = InferenceResult(
        prior_samples=prior,
        posterior_samples=posterior,
        losses=[1.0, 0.5],
        num_samples=50,
    )
    elements = build_cytoscape_elements(ditto_graph, result, CONFIG)
    nodes = {e["data"]["id"]: e["data"] for e in elements if "source" not in e["data"]}

    mu_data = nodes["mu"]
    assert mu_data["tag"] == "latent"
    assert mu_data["image_prior"], "latent node should have a prior thumbnail"
    assert mu_data["image_posterior"], "latent node should have a posterior thumbnail"
    # The two thumbnails should be different images (different titles/colors).
    assert mu_data["image_prior"] != mu_data["image_posterior"]


def test_build_cytoscape_elements_latent_without_posterior_falls_back():
    """If posterior samples aren't available for a latent node, only the
    prior thumbnail is populated."""
    variables = [AnnotatedVariable("mu", "latent", 1, "dist.Normal(0., 1.)")]
    ditto_graph = build_graph(variables)
    result = InferenceResult(
        prior_samples={"mu": torch.randn(50)},
        posterior_samples={},
        losses=[],
        num_samples=50,
    )
    elements = build_cytoscape_elements(ditto_graph, result, CONFIG)
    mu_data = [e["data"] for e in elements if e["data"].get("id") == "mu"][0]
    assert mu_data["image_prior"]
    assert mu_data["image_posterior"] == ""
