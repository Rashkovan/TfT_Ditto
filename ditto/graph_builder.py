"""Build a :class:`DittoGraph` from a list of :class:`AnnotatedVariable`.

Edges are inferred from textual references: a variable's RHS expression is
parsed and every ``ast.Name`` found is intersected with the set of *known*
annotated variable names. Names of library functions (``torch``, ``dist``,
etc.) drop out naturally because they are not annotated.

When a source ``filepath`` is provided, unannotated intermediate assignments
(e.g. ``mu_k = torch.sigmoid(3.0 - 4.0 * stress)``) are also parsed and
used to resolve transitive dependencies. Without this, a chain like
``stress → mu_k → knowledge`` would appear as two disconnected edges because
``mu_k`` is not in the annotated-variable set.
"""
from __future__ import annotations

import ast
from typing import Dict, List, Optional, Set

import networkx as nx

from ditto.models import AnnotatedVariable, DittoGraph


def extract_names(expression: str) -> Set[str]:
    """Return all bare ``ast.Name`` identifiers referenced in ``expression``.

    The expression is parsed in expression mode so it must be a valid Python
    expression (which is exactly what ``ast.unparse(node.value)`` produces).
    """
    tree = ast.parse(expression, mode="eval")
    names: Set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            names.add(node.id)
    return names


def _collect_intermediates(filepath: str) -> Dict[str, Set[str]]:
    """Return a mapping of every unannotated assignment name to the names it references.

    Covers both ``ast.Assign`` (single bare-name target) and ``ast.AnnAssign``
    nodes anywhere in the file, including inside function bodies.
    """
    with open(filepath, "r", encoding="utf-8") as fh:
        source = fh.read()
    tree = ast.parse(source)
    intermediates: Dict[str, Set[str]] = {}
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


def _resolve_deps(
    names: Set[str],
    known_names: Set[str],
    intermediates: Dict[str, Set[str]],
    visited: Optional[Set[str]] = None,
) -> Set[str]:
    """Transitively resolve a set of names to the annotated variables they reach.

    For each name: if it is annotated, add it directly. If it is an unannotated
    intermediate, recurse into its own references. Cycles in unannotated
    intermediates are broken by the ``visited`` guard.
    """
    if visited is None:
        visited = set()
    result: Set[str] = set()
    for name in names:
        if name in visited:
            continue
        visited.add(name)
        if name in known_names:
            result.add(name)
        elif name in intermediates:
            result |= _resolve_deps(
                intermediates[name], known_names, intermediates, visited
            )
    return result


def build_graph(
    variables: List[AnnotatedVariable],
    filepath: Optional[str] = None,
) -> DittoGraph:
    """Build a DAG over annotated variables.

    The dependency rule is::

        deps = resolve(referenced_names, known_annotated_names) - {self_name}

    When ``filepath`` is supplied, unannotated intermediate assignments are
    collected from the source and used to resolve transitive dependencies so
    that chains like ``stress → mu_k → knowledge`` produce a direct edge
    ``stress → knowledge`` in the graph.

    Raises
    ------
    ValueError
        If the resulting graph contains a cycle.
    """
    known_names = {v.name for v in variables}
    intermediates = _collect_intermediates(filepath) if filepath else {}

    # Remove annotated names from intermediates so they are never treated as
    # pass-through nodes; they always terminate the transitive search.
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
        raise ValueError(
            f"Ditto annotations form a cycle: {cycle}. The dependency graph "
            f"must be a DAG."
        )

    return DittoGraph(graph=graph, variables=var_map)
