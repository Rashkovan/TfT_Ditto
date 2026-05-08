"""Tests for :mod:`ditto.inference_engine`.

SVI runs use a tiny ``svi_steps`` value so the suite stays fast.
"""
from __future__ import annotations

import os

import torch

from ditto.graph_builder import build_graph
from ditto.inference_engine import (
    extract_distribution_from_sample_call,
    extract_observed_data,
    is_pyro_sample_call,
    load_user_module,
    run_inference,
    sample_prior,
)
from ditto.models import AnnotatedVariable
from ditto.parser import parse_file

FIXTURES = os.path.join(os.path.dirname(__file__), "fixtures")

SMALL_CONFIG = {
    "inference": {
        "svi_steps": 50,
        "learning_rate": 0.05,
        "num_samples": 32,
    },
    "visualization": {},
    "server": {"port": 8050, "debug": False},
    "export": {"path": None, "format": "png", "dpi": 100},
}


def test_sample_prior_returns_tensor_of_correct_shape():
    var = AnnotatedVariable(
        name="mu",
        tag="prior",
        line_number=1,
        raw_expression="dist.Normal(0., 1.)",
    )
    samples = sample_prior(var, num_samples=64)
    assert isinstance(samples, torch.Tensor)
    assert samples.shape[0] == 64
    # Detached: should not have a grad_fn.
    assert samples.grad_fn is None


def test_run_inference_with_simple_model_returns_prior_and_posterior():
    filepath = os.path.join(FIXTURES, "simple_model.py")
    variables = parse_file(filepath)
    ditto_graph = build_graph(variables)
    user_module = load_user_module(filepath)

    result = run_inference(ditto_graph, user_module, SMALL_CONFIG)

    # `mu` is latent: it should have BOTH a prior draw and a posterior draw.
    assert "mu" in result.prior_samples
    assert result.prior_samples["mu"].shape[0] == SMALL_CONFIG["inference"]["num_samples"]
    assert "mu" in result.posterior_samples

    # SVI ran (latent var present), so we recorded losses.
    assert len(result.losses) == SMALL_CONFIG["inference"]["svi_steps"]
    assert result.num_samples == SMALL_CONFIG["inference"]["num_samples"]

    # `x` is observed: posterior predictive should be present.
    assert "x" in result.posterior_samples


def test_run_inference_without_latent_skips_svi(tmp_path):
    # Construct a tiny module that has only priors (no `latent`).
    src = (
        "import pyro.distributions as dist\n"
        "\n"
        "# !Ditto: prior\n"
        "a = dist.Normal(0., 1.)\n"
        "\n"
        "# !Ditto: prior\n"
        "b = dist.Normal(0., 1.)\n"
    )
    path = tmp_path / "priors_only.py"
    path.write_text(src)
    variables = parse_file(str(path))
    ditto_graph = build_graph(variables)
    user_module = load_user_module(str(path))

    result = run_inference(ditto_graph, user_module, SMALL_CONFIG)
    assert set(result.prior_samples.keys()) == {"a", "b"}
    assert result.posterior_samples == {}
    assert result.losses == []


def test_is_pyro_sample_call_detects_sample_calls():
    assert is_pyro_sample_call('pyro.sample("x", dist.Normal(0., 1.))') is True
    assert is_pyro_sample_call('sample("x", dist.Normal(0., 1.))') is True
    assert is_pyro_sample_call("dist.Normal(0., 1.)") is False
    # Non-call expression and syntactically broken input should both return
    # False rather than blowing up.
    assert is_pyro_sample_call("42") is False
    assert is_pyro_sample_call("(((") is False


def test_extract_distribution_from_sample_call_returns_second_arg():
    expr = 'pyro.sample("lambda_plus", dist.Gamma(1.0, 1.0))'
    assert extract_distribution_from_sample_call(expr) == "dist.Gamma(1.0, 1.0)"


def test_sample_prior_handles_pyro_sample_call_expression():
    var = AnnotatedVariable(
        name="lambda_plus",
        tag="latent",
        line_number=1,
        raw_expression='pyro.sample("lambda_plus", dist.Gamma(1.0, 1.0))',
        is_sample_call=True,
    )
    samples = sample_prior(var, num_samples=16)
    assert isinstance(samples, torch.Tensor)
    assert samples.shape[0] == 16
    # Gamma(1, 1) draws are strictly positive.
    assert torch.all(samples > 0)


def test_extract_observed_data_returns_observed_site_values():
    """A model with an ``obs=`` site should expose the conditioned tensor."""
    import pyro
    import pyro.distributions as dist

    expected = torch.tensor([0.5, 1.5, 2.5])

    def model(data):
        mu = pyro.sample("mu", dist.Normal(0.0, 1.0))
        with pyro.plate("d", data.shape[0]):
            pyro.sample("obs", dist.Normal(mu, 1.0), obs=data)

    observed = extract_observed_data(model, (expected,), {})
    assert "obs" in observed
    assert torch.equal(observed["obs"], expected)
    # The latent ``mu`` site is sampled, not observed, so it must not appear.
    assert "mu" not in observed


def test_inference_result_samples_property_merges():
    """Backward-compat: ``.samples`` exposes a single merged mapping."""
    from ditto.models import InferenceResult
    pri = {"a": torch.zeros(4), "b": torch.zeros(4)}
    post = {"b": torch.ones(4), "c": torch.ones(4)}
    result = InferenceResult(prior_samples=pri, posterior_samples=post,
                             losses=[], num_samples=4)
    merged = result.samples
    assert set(merged.keys()) == {"a", "b", "c"}
    # Posterior wins for overlapping site names.
    assert torch.equal(merged["b"], torch.ones(4))
