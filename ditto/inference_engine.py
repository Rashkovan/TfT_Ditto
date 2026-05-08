"""Run prior sampling and SVI for an annotated user module.

The engine evaluates ``prior`` and ``latent`` expressions in a small namespace
exposing ``torch``, ``pyro`` and ``dist`` to produce prior draws. When any
``latent`` variables are declared, Ditto auto-creates a guide over the user's
``model`` callable (``AutoNormal`` for bare-distribution latents, or
``AutoNormalizingFlow`` if any latent is declared via ``pyro.sample(...)``),
runs SVI, and then uses ``Predictive`` to collect posterior draws for
``latent`` and ``observed`` sites.

Annotated ``latent``/``prior`` expressions may either be a bare distribution
(``dist.Normal(0., 1.)``) or a ``pyro.sample("name", dist.X(...))`` call —
the latter is common when annotations live inside the user's ``model``
function. ``sample_prior`` extracts the underlying distribution in that case
so it can draw raw prior samples instead of executing the sample site.
"""
from __future__ import annotations

import ast
import importlib.util
import sys
import uuid
from types import ModuleType
from typing import List, Tuple

import pyro
import pyro.distributions as dist
import pyro.optim
import torch
from pyro import poutine
from pyro.distributions.transforms import block_autoregressive
from pyro.infer import SVI, Predictive, Trace_ELBO
from pyro.infer.autoguide import AutoNormal, AutoNormalizingFlow

from ditto.models import AnnotatedVariable, DittoGraph, InferenceResult

#: Namespace used to ``eval`` prior RHS expressions. Kept tiny on purpose so
#: users don't accidentally rely on globals from the engine module.
EVAL_NAMESPACE = {"torch": torch, "pyro": pyro, "dist": dist}


def is_pyro_sample_call(expression: str) -> bool:
    """Return True if ``expression`` is a top-level ``(pyro.)sample(...)`` call.

    Detects both ``sample("x", dist.Normal(0., 1.))`` (bare ``ast.Name``) and
    ``pyro.sample("x", dist.Normal(0., 1.))`` (``ast.Attribute``). Any other
    expression — including bare distribution constructors — returns False.
    Returns False on syntactically invalid input rather than raising, so
    callers can use it as a cheap predicate.
    """
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError:
        return False
    node = tree.body
    if not isinstance(node, ast.Call):
        return False
    func = node.func
    if isinstance(func, ast.Name) and func.id == "sample":
        return True
    if isinstance(func, ast.Attribute) and func.attr == "sample":
        return True
    return False


def _get_pyro_site_name(expression: str) -> str:
    """Extract the string-literal site name from a ``pyro.sample(name, ...)`` call."""
    tree = ast.parse(expression, mode="eval")
    node = tree.body
    if not isinstance(node, ast.Call) or not node.args:
        raise ValueError(
            f"Expected a pyro.sample(name, ...) call, got: {expression!r}"
        )
    first = node.args[0]
    if not isinstance(first, ast.Constant) or not isinstance(first.value, str):
        raise ValueError(
            f"First argument to pyro.sample must be a string literal, "
            f"got {type(first).__name__!r} in: {expression!r}"
        )
    return first.value


def extract_distribution_from_sample_call(expression: str) -> str:
    """Return the distribution sub-expression from a ``pyro.sample(...)`` call.

    Assumes ``expression`` is a ``Call`` whose second positional argument
    (index 1) is the distribution. Raises ``ValueError`` if the call has fewer
    than two positional arguments.
    """
    tree = ast.parse(expression, mode="eval")
    node = tree.body
    if not isinstance(node, ast.Call) or len(node.args) < 2:
        raise ValueError(
            f"Expected a pyro.sample(name, distribution, ...) call with at "
            f"least two positional arguments, got: {expression!r}"
        )
    return ast.unparse(node.args[1])


def load_user_module(filepath: str) -> ModuleType:
    """Import ``filepath`` as a fresh Python module.

    A unique module name is generated to avoid colliding with anything already
    in ``sys.modules`` (notably useful when re-running inference in tests).
    """
    module_name = f"_ditto_user_{uuid.uuid4().hex}"
    spec = importlib.util.spec_from_file_location(module_name, filepath)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load Python module from {filepath!r}.")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def sample_prior(variable: AnnotatedVariable, num_samples: int) -> torch.Tensor:
    """Draw ``num_samples`` from a variable's declared prior distribution.

    The variable's ``raw_expression`` is evaluated to produce a Pyro/Torch
    distribution object, which is then sampled. Returned tensor is detached
    so downstream consumers don't accidentally retain autograd state.
    """
    # When the user's annotated expression is a ``pyro.sample(...)`` call
    # (typical inside a model function), evaluating it directly would execute
    # the sample and return a tensor. Extract the underlying distribution
    # sub-expression instead so we can draw raw prior samples.
    if variable.is_sample_call or is_pyro_sample_call(variable.raw_expression):
        expression_to_eval = extract_distribution_from_sample_call(
            variable.raw_expression
        )
    else:
        expression_to_eval = variable.raw_expression

    # Pass a fresh copy so eval's implicit ``__builtins__`` insertion does not
    # mutate the module-level namespace across calls.
    distribution = eval(expression_to_eval, dict(EVAL_NAMESPACE))  # noqa: S307
    if not hasattr(distribution, "sample"):
        raise TypeError(
            f"Prior expression for '{variable.name}' did not evaluate to a "
            f"distribution (got {type(distribution).__name__})."
        )
    samples = distribution.sample((num_samples,))
    return samples.detach()


def run_svi(
    model,
    guide,
    model_args: tuple,
    model_kwargs: dict,
    svi_steps: int,
    learning_rate: float,
) -> List[float]:
    """Run Stochastic Variational Inference for ``svi_steps`` iterations.

    Uses ``pyro.optim.Adam`` (the Pyro wrapper, *not* ``torch.optim.Adam``;
    the latter does not implement Pyro's optimizer interface) and the standard
    ``Trace_ELBO`` loss. Clears the Pyro param store before starting so that
    repeated calls (e.g. in tests) start from a clean slate.
    """
    pyro.clear_param_store()
    optimizer = pyro.optim.Adam({"lr": learning_rate})
    svi = SVI(model, guide, optimizer, loss=Trace_ELBO())

    losses: List[float] = []
    for step in range(svi_steps):
        loss = svi.step(*model_args, **model_kwargs)
        losses.append(float(loss))
        if step % 200 == 0:
            print(f"[SVI] step {step:>5d}  loss = {loss:.4f}")
    return losses


def collect_posterior_samples(
    model,
    guide,
    model_args: tuple,
    model_kwargs: dict,
    num_samples: int,
    observed_sites=None,
):
    """Draw posterior predictive samples using ``Predictive``.

    The model is wrapped in ``pyro.poutine.uncondition`` so every internal
    ``pyro.sample(..., obs=...)`` call is converted into a fresh sample site.
    This is more robust than stripping kwargs because it works even when the
    observed data is constructed *inside* the model rather than passed in.
    The original ``model_args`` / ``model_kwargs`` are forwarded unchanged so
    plate sizes and other shape-determining inputs remain correct.

    ``observed_sites`` may be passed as a ``return_sites`` filter; ``None``
    (the default) means all sites are returned.
    """
    unconditioned = poutine.uncondition(model)
    predictive = Predictive(
        unconditioned,
        guide=guide,
        num_samples=num_samples,
        return_sites=observed_sites,
    )
    raw = predictive(*model_args, **model_kwargs)
    return {name: tensor.detach() for name, tensor in raw.items()}


def find_discrete_latent_sites(model, model_args: tuple, model_kwargs: dict) -> set:
    """Return names of unobserved sample sites whose distribution is discrete.

    Discrete sites cannot be handled by continuous autoguides (AutoNormal,
    AutoNormalizingFlow). Callers should hide them via ``poutine.block`` before
    constructing the guide.
    """
    trace = poutine.trace(model).get_trace(*model_args, **model_kwargs)
    return {
        name
        for name, node in trace.nodes.items()
        if node.get("type") == "sample"
        and not node.get("is_observed", False)
        and getattr(node.get("fn"), "has_enumerate_support", False)
    }


def extract_observed_data(model, model_args: tuple, model_kwargs: dict):
    """Run ``model`` once under a Pyro trace and return observed site values.

    Returns a mapping from site name to the conditioned ``value`` tensor for
    every ``pyro.sample`` site flagged ``is_observed`` in the trace. Used to
    surface the actual data alongside posterior predictive draws in the
    visualizer.
    """
    trace = poutine.trace(model).get_trace(*model_args, **model_kwargs)
    return {
        name: node["value"].detach()
        for name, node in trace.nodes.items()
        if node.get("type") == "sample" and node.get("is_observed", False)
    }


def get_model_args(user_module: ModuleType, config: dict) -> Tuple[tuple, dict]:
    """Resolve ``(args, kwargs)`` to pass into the user's ``model`` callable.

    The user must define ``get_data() -> (args, kwargs)`` in their source file.
    A helpful ``AttributeError`` is raised otherwise so they don't have to
    decode a generic ``module has no attribute`` message.
    """
    if not hasattr(user_module, "get_data"):
        raise AttributeError(
            "User module must define a `get_data()` function returning "
            "`(args, kwargs)` to pass into `model`. Example:\n\n"
            "    def get_data():\n"
            "        return (), {'obs': torch.tensor([0.5, 1.0])}\n"
        )
    args, kwargs = user_module.get_data()
    return args, kwargs


def _strip_observed(model_kwargs: dict) -> dict:
    """Return a copy of ``model_kwargs`` with observed-data keys removed.

    By convention, observed values are passed under the key ``obs``. Removing
    it lets ``Predictive`` re-sample the corresponding observation site.
    """
    stripped = dict(model_kwargs)
    stripped.pop("obs", None)
    return stripped


def run_inference(
    ditto_graph: DittoGraph,
    user_module: ModuleType,
    config: dict,
) -> InferenceResult:
    """Top-level driver: sample priors, optionally run SVI, return results.

    Behavior:

    * Every ``prior`` and ``latent`` variable is sampled from its declared
      distribution to populate ``prior_samples``. Latent variables thus get
      *both* a prior draw (for the prior plot) and, downstream, a posterior
      draw (for the posterior plot).
    * If any ``latent`` variables are present, Ditto constructs a guide over
      ``user_module.model`` automatically — ``AutoNormal`` when every latent
      is a bare distribution expression, otherwise ``AutoNormalizingFlow``
      when any latent is declared via ``pyro.sample(...)``. SVI runs against
      the user's ``model``, then ``Predictive`` (wrapping the model in
      ``poutine.uncondition``) draws posterior samples for every site,
      including observed ones. The actual observed data is then concatenated
      onto observed-site tensors so the visualizer can render both together.
    """
    inference_cfg = config["inference"]
    num_samples = inference_cfg["num_samples"]
    svi_steps = inference_cfg["svi_steps"]
    learning_rate = inference_cfg["learning_rate"]

    prior_samples: dict = {}
    posterior_samples: dict = {}

    # 1) Sample every prior- and latent-tagged variable up front. For latent
    #    variables this yields the "prior" plot in the visualizer; the
    #    posterior plot comes from SVI + Predictive below.
    #
    #    Variables whose annotated expression is a pyro.sample(...) call may
    #    reference intermediate names that only exist in the model's execution
    #    scope (e.g. `dist.Beta(mu_k * kappa, ...)`). Those cannot be eval'd
    #    in isolation, so they are collected here and sampled together via
    #    Pyro's prior predictive in the second pass below.
    sample_call_prior_latents: list = []
    for name in ditto_graph.topological_order():
        var = ditto_graph.variables[name]
        if var.tag in ("prior", "latent"):
            if var.is_sample_call or is_pyro_sample_call(var.raw_expression):
                sample_call_prior_latents.append(var)
            else:
                prior_samples[name] = sample_prior(var, num_samples)

    # 1b) Prior predictive pass for pyro.sample(...) annotated variables.
    #     Predictive(model, num_samples=N) with no guide runs the model
    #     forward under the joint prior, correctly resolving inter-variable
    #     dependencies that eval in isolation cannot handle.
    if sample_call_prior_latents:
        if not hasattr(user_module, "model"):
            raise AttributeError(
                "User module must define a `model` callable when any annotated "
                "variable uses pyro.sample(...)."
            )
        _pp_args, _pp_kwargs = get_model_args(user_module, config)
        prior_predictive = Predictive(user_module.model, num_samples=num_samples)
        prior_draws = prior_predictive(*_pp_args, **_pp_kwargs)
        for var in sample_call_prior_latents:
            site_name = _get_pyro_site_name(var.raw_expression)
            tensor = prior_draws.get(site_name)
            if tensor is not None:
                prior_samples[var.name] = tensor.detach()

    losses: List[float] = []

    # 2) If the user declared any `latent` variables, auto-create a guide
    #    over their `model` (AutoNormal for bare-distribution latents,
    #    AutoNormalizingFlow if any latent is a pyro.sample(...) call), run
    #    SVI, and harvest posterior samples for latent + observed sites.
    latent_vars = [
        v for v in ditto_graph.variables.values() if v.tag == "latent"
    ]
    if latent_vars:
        if not hasattr(user_module, "model"):
            raise AttributeError(
                "User module must define a `model` callable when any 'latent' "
                "variable is annotated."
            )

        model = user_module.model
        args, kwargs = get_model_args(user_module, config)

        # Discrete latent sites (Bernoulli, Categorical, …) cannot be handled
        # by continuous autoguides. Block them so the guide only covers the
        # continuous latents; discrete sites will display prior samples only.
        discrete_sites = find_discrete_latent_sites(model, args, kwargs)
        guide_model = (
            poutine.block(model, hide=list(discrete_sites))
            if discrete_sites
            else model
        )

        any_sample_call_latent = any(
            v.is_sample_call or is_pyro_sample_call(v.raw_expression)
            for v in latent_vars
        )
        if any_sample_call_latent:
            guide = AutoNormalizingFlow(guide_model, block_autoregressive)
        else:
            guide = AutoNormal(guide_model)

        losses = run_svi(
            model=model,
            guide=guide,
            model_args=args,
            model_kwargs=kwargs,
            svi_steps=svi_steps,
            learning_rate=learning_rate,
        )

        # ``poutine.uncondition`` (inside ``collect_posterior_samples``)
        # handles observed sites — both kwarg-passed and model-internal — so
        # the original args/kwargs are forwarded unchanged.
        posterior_samples = collect_posterior_samples(
            model=model,
            guide=guide,
            model_args=args,
            model_kwargs=kwargs,
            num_samples=num_samples,
        )

        # Surface the actual observed data alongside posterior predictive
        # draws for any observed site. Concatenating along dim 0 lets the
        # visualizer show both the data and the predictive distribution as a
        # single marginal KDE.
        observed_values = extract_observed_data(model, args, kwargs)
        observed_var_names = {
            v.name for v in ditto_graph.variables.values() if v.tag == "observed"
        }
        for site_name, obs_tensor in observed_values.items():
            if site_name not in observed_var_names:
                continue
            existing = posterior_samples.get(site_name)
            if existing is None:
                posterior_samples[site_name] = obs_tensor
                continue
            # Broadcast the observed tensor up to a leading sample dim so it
            # can be concatenated with the predictive draws.
            obs_expanded = obs_tensor.unsqueeze(0)
            try:
                posterior_samples[site_name] = torch.cat(
                    [existing, obs_expanded], dim=0
                )
            except RuntimeError:
                # Shape mismatch — fall back to keeping the predictive draws
                # alone rather than failing the whole inference run.
                pass

    return InferenceResult(
        prior_samples=prior_samples,
        posterior_samples=posterior_samples,
        losses=losses,
        num_samples=num_samples,
    )
