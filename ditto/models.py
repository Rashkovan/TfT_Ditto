"""Core dataclasses used throughout Ditto.

These are intentionally dependency-light (only ``networkx`` and ``torch``)
so every other ditto module can import from here freely without creating
import cycles.
"""
from dataclasses import dataclass, field
from typing import Dict, FrozenSet, List

import networkx as nx
import torch


@dataclass
class AnnotatedVariable:
    """A single ``# !Ditto: <tag>`` annotated assignment in user source.

    Attributes
    ----------
    name:
        The Python identifier on the left-hand side of the assignment.
    tag:
        One of ``"prior"``, ``"latent"``, ``"observed"``.
    line_number:
        1-indexed line number of the annotation comment in the source file.
    raw_expression:
        The right-hand side of the assignment, recovered via ``ast.unparse``.
    dependencies:
        Names of *other annotated* variables referenced by ``raw_expression``.
        Filled in by :mod:`ditto.graph_builder` after parsing.
    is_sample_call:
        True if ``raw_expression`` is a top-level ``pyro.sample(...)`` call
        (or a bare ``sample(...)`` call). Used by the inference engine to
        extract the underlying distribution rather than executing the sample
        and getting a tensor back.
    """

    name: str
    tag: str  # primary tag for colors and CSS selectors
    line_number: int
    raw_expression: str
    dependencies: List[str] = field(default_factory=list)
    is_sample_call: bool = False
    tags: FrozenSet[str] = field(default_factory=frozenset)  # full set, may include multiple


@dataclass
class DittoGraph:
    """Wraps a ``networkx.DiGraph`` together with the variable metadata.

    The graph nodes are variable names (strings); edges go ``dep -> variable``
    so a topological sort yields evaluation order.
    """

    graph: nx.DiGraph
    variables: Dict[str, AnnotatedVariable]

    def topological_order(self) -> List[str]:
        """Return variable names in topological (dependency-respecting) order."""
        return list(nx.topological_sort(self.graph))

    def predecessors(self, name: str) -> List[str]:
        """Return the immediate dependencies of ``name``."""
        return list(self.graph.predecessors(name))


@dataclass
class InferenceResult:
    """Container for the output of :func:`ditto.inference_engine.run_inference`.

    Attributes
    ----------
    prior_samples:
        Mapping from variable name to a tensor of prior draws. Populated for
        ``prior``-tagged variables and (additionally) for ``latent``-tagged
        variables, since latent vars are displayed with both prior and
        posterior plots.
    posterior_samples:
        Mapping from site name to a tensor of posterior draws (from
        ``Predictive`` over the trained guide). Populated for ``latent`` and
        ``observed`` sites whenever any ``latent`` variables are present.
    losses:
        Per-step ELBO losses recorded during SVI. Empty list if no ``latent``
        variable was present.
    num_samples:
        Number of draws stored per variable (the leading tensor dim).
    """

    prior_samples: Dict[str, torch.Tensor]
    posterior_samples: Dict[str, torch.Tensor]
    losses: List[float]
    num_samples: int

    @property
    def samples(self) -> Dict[str, torch.Tensor]:
        """Backward-compatible merged view: posterior overrides prior per site.

        Maintained so older callers can still access ``result.samples`` and
        get a single mapping per variable name.
        """
        merged: Dict[str, torch.Tensor] = dict(self.prior_samples)
        for name, tensor in self.posterior_samples.items():
            merged[name] = tensor
        return merged
