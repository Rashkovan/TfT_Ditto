"""Tests for :mod:`ditto.graph_builder`."""
from __future__ import annotations

import pytest

from ditto.graph_builder import build_graph, extract_names
from ditto.models import AnnotatedVariable


def _var(name: str, expr: str, tag: str = "prior") -> AnnotatedVariable:
    return AnnotatedVariable(name=name, tag=tag, line_number=1, raw_expression=expr)


def test_extract_names_finds_bare_identifiers():
    names = extract_names("a + b * torch.tensor(c)")
    # Includes attribute owner and bare names; library names will simply not
    # match any annotated variable downstream.
    assert {"a", "b", "torch", "c"}.issubset(names)


def test_edges_created_for_known_dependencies():
    variables = [
        _var("a", "dist.Normal(0., 1.)"),
        _var("b", "dist.Normal(0., 1.)"),
        _var("c", "a + b", tag="observed"),
    ]
    graph = build_graph(variables)
    assert ("a", "c") in graph.graph.edges()
    assert ("b", "c") in graph.graph.edges()
    assert graph.variables["c"].dependencies == ["a", "b"]


def test_no_edges_for_unrelated_variables():
    variables = [
        _var("a", "dist.Normal(0., 1.)"),
        _var("b", "dist.Normal(0., 1.)"),
    ]
    graph = build_graph(variables)
    assert graph.graph.number_of_edges() == 0


def test_cycle_detection_raises_value_error():
    variables = [
        _var("a", "b + 1"),
        _var("b", "a + 1"),
    ]
    with pytest.raises(ValueError, match="cycle"):
        build_graph(variables)


def test_self_reference_excluded_from_dependencies():
    # Pathological but valid: a variable's expression mentions its own name.
    variables = [_var("a", "a + 1")]
    graph = build_graph(variables)
    # No self-loop, no edges at all, no cycle error.
    assert list(graph.graph.edges()) == []
    assert graph.variables["a"].dependencies == []


def test_topological_order_respects_dependencies():
    variables = [
        _var("a", "dist.Normal(0., 1.)"),
        _var("b", "a + 1"),
        _var("c", "b + a", tag="observed"),
    ]
    graph = build_graph(variables)
    order = graph.topological_order()
    assert order.index("a") < order.index("b") < order.index("c")
