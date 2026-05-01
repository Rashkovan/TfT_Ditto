import importlib.util
import sys
from pathlib import Path
from typing import Dict, List

import pyro
import pyro.distributions as dist
import torch
from pyro.infer import SVI, Trace_ELBO
from pyro.infer import Predictive

from models import AnnotatedVariable

N_SAMPLES = 500


def sample_variables(
    variables: List[AnnotatedVariable],
    model_path: Path,
    svi_steps: int = 1000,
) -> Dict[str, List[float]]:
    """Return {variable_name: [float, ...]} for every annotated variable."""
    results: Dict[str, List[float]] = {}

    prior_vars = [v for v in variables if v.tag == "prior"]
    svi_vars = [v for v in variables if v.tag in ("approx", "observed")]

    # --- Prior sampling ---
    for var in prior_vars:
        try:
            samples = _sample_prior(var.expr)
            results[var.name] = samples
        except Exception as e:
            results[var.name] = []
            print(f"[ditto] prior sampling failed for {var.name!r}: {e}")

    # --- SVI / posterior sampling ---
    if svi_vars:
        try:
            posterior_samples = _sample_posterior(model_path, svi_vars, svi_steps)
            results.update(posterior_samples)
        except Exception as e:
            print(f"[ditto] SVI failed: {e}")
            for var in svi_vars:
                results.setdefault(var.name, [])

    return results


def _sample_prior(expr: str) -> List[float]:
    namespace = {
        "pyro": pyro,
        "dist": dist,
        "torch": torch,
    }
    distribution = eval(expr, namespace)  # noqa: S307 — controlled namespace
    samples = distribution.sample((N_SAMPLES,))
    return _to_flat_list(samples)


def _sample_posterior(
    model_path: Path,
    svi_vars: List[AnnotatedVariable],
    svi_steps: int,
) -> Dict[str, List[float]]:
    pyro.clear_param_store()

    # Load user module
    module_name = model_path.stem
    spec = importlib.util.spec_from_file_location(module_name, str(model_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module from {model_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)

    # Expect the module to expose `model` and `guide` callables
    if not hasattr(module, "model"):
        raise AttributeError("Model file must define a 'model' function for SVI.")
    if not hasattr(module, "guide"):
        raise AttributeError("Model file must define a 'guide' function for SVI.")

    model_fn = module.model
    guide_fn = module.guide

    optimizer = pyro.optim.Adam({"lr": 0.01})
    svi = SVI(model_fn, guide_fn, optimizer, loss=Trace_ELBO())

    for _ in range(svi_steps):
        svi.step()

    # Draw posterior predictive samples (no observed conditioning)
    predictive = Predictive(model_fn, guide=guide_fn, num_samples=N_SAMPLES)
    samples_dict = predictive()

    results: Dict[str, List[float]] = {}
    var_names = {v.name for v in svi_vars}
    for name, tensor in samples_dict.items():
        if name in var_names:
            results[name] = _to_flat_list(tensor)

    return results


def _to_flat_list(tensor: torch.Tensor) -> List[float]:
    return tensor.detach().flatten().tolist()
