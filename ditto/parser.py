"""Parse user source files for ``# !Ditto:`` annotations.

The parser is deliberately two-pass:

1. A regex scan finds every ``# !Ditto: <tag>`` comment line and records
   ``(line_number, tag)``. We *must* use a regex here because Python's ``ast``
   module strips comments entirely, so AST traversal alone cannot see them.
2. An AST walk finds every ``ast.Assign`` / ``ast.AnnAssign`` statement, plus
   ``ast.Expr`` statements that wrap a bare ``(pyro.)sample(...)`` call. For
   each recorded annotation at line ``L`` we associate it with the *closest*
   such statement whose ``lineno > L`` (so blank lines between the comment and
   the statement are tolerated). Bare ``pyro.sample("name", ...)`` statements
   take their site name from the string-literal first argument.
"""
from __future__ import annotations

import ast
import re
from typing import List, Tuple

from ditto.models import AnnotatedVariable

# Imported lazily inside ``parse_file`` would create a cycle; instead we keep
# the predicate logic local here to avoid importing the inference engine just
# to classify an expression. This mirrors ``inference_engine.is_pyro_sample_call``.


def _is_pyro_sample_call_ast(value_node: ast.AST) -> bool:
    """Return True if ``value_node`` is a ``(pyro.)sample(...)`` ``Call``."""
    if not isinstance(value_node, ast.Call):
        return False
    func = value_node.func
    if isinstance(func, ast.Name) and func.id == "sample":
        return True
    if isinstance(func, ast.Attribute) and func.attr == "sample":
        return True
    return False


#: Recognized annotation tags. Anything else triggers a ``ValueError``.
VALID_TAGS = {"prior", "observed", "latent"}

#: Regex matching a Ditto annotation comment. Captures everything after the
#: colon so comma-separated multi-tags (e.g. "prior, latent") are preserved.
_ANNOTATION_RE = re.compile(r"#\s*!Ditto:\s*(.+)")


def _parse_tags(raw: str) -> frozenset:
    """Return the frozenset of individual tags from a raw annotation string.

    ``"prior, latent"`` → ``frozenset({"prior", "latent"})``.
    """
    return frozenset(t.strip() for t in raw.split(","))


def _primary_tag(tags: frozenset) -> str:
    """Return the single display tag used for node colors and CSS selectors.

    Priority order: ``latent`` > ``observed`` > ``prior``. Choosing ``latent``
    first means a ``prior, latent`` node inherits the green latent colour.
    """
    for t in ("latent", "observed", "prior"):
        if t in tags:
            return t
    return next(iter(tags))  # fallback so callers get something to validate


def find_annotation_lines(source: str) -> List[Tuple[int, str]]:
    """Return ``[(line_number, tag), ...]`` for every Ditto comment.

    Line numbers are 1-indexed to match Python AST conventions.
    """
    annotations: List[Tuple[int, str]] = []
    for idx, line in enumerate(source.splitlines(), start=1):
        match = _ANNOTATION_RE.search(line)
        if match:
            annotations.append((idx, match.group(1).strip()))
    return annotations


def _find_assignment_after(tree: ast.AST, line: int):
    """Return the closest annotatable statement after ``line``.

    Considers ``ast.Assign`` / ``ast.AnnAssign`` nodes as well as ``ast.Expr``
    nodes whose ``.value`` is a ``(pyro.)sample(...)`` call. The latter lets
    Ditto annotate bare ``pyro.sample("name", dist, obs=...)`` expression
    statements that don't bind a Python name (typical of conditioning sites
    inside a model function).

    Returns ``None`` if no such node exists.
    """
    candidates = []
    for node in ast.walk(tree):
        if not hasattr(node, "lineno") or node.lineno <= line:
            continue
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            candidates.append(node)
        elif isinstance(node, ast.Expr) and _is_pyro_sample_call_ast(node.value):
            candidates.append(node)
    if not candidates:
        return None
    # The closest annotatable statement after the comment is the one with the
    # smallest ``lineno``. ``ast.walk`` does not guarantee order, so sort
    # explicitly. When an Assign and an Expr(pyro.sample(...)) tie on lineno
    # (which they shouldn't in practice), assignments sort earlier because
    # ``isinstance`` check above doesn't change ordering — ties are broken by
    # the original walk order, which is fine.
    candidates.sort(key=lambda n: n.lineno)
    return candidates[0]


def _extract_sample_site_name(call_node: ast.Call, lineno: int) -> str:
    """Return the string-literal site name from a ``pyro.sample(name, ...)`` call.

    Raises ``ValueError`` if the first positional argument is missing or is
    not a string literal — Ditto needs a stable site name to key results.
    """
    if not call_node.args:
        raise ValueError(
            f"pyro.sample(...) at line {lineno} requires a site name as its "
            f"first positional argument."
        )
    first = call_node.args[0]
    if not (isinstance(first, ast.Constant) and isinstance(first.value, str)):
        raise ValueError(
            f"Ditto requires the first argument to pyro.sample(...) to be a "
            f"string literal (got {type(first).__name__}) at line {lineno}."
        )
    return first.value


def _extract_target_name(node) -> str:
    """Return the LHS identifier of a (Ann)Assign, or raise ``ValueError``.

    Tuple unpacking and attribute targets are not supported because each
    annotated variable is meant to correspond to a single named random variable
    in the resulting DAG.
    """
    if isinstance(node, ast.AnnAssign):
        target = node.target
    else:  # ast.Assign
        if len(node.targets) != 1:
            raise ValueError(
                f"Ditto annotations require a single assignment target "
                f"(line {node.lineno})."
            )
        target = node.targets[0]
    if not isinstance(target, ast.Name):
        raise ValueError(
            f"Ditto annotations only support simple name targets, got "
            f"{type(target).__name__} at line {node.lineno}."
        )
    return target.id


def parse_file(filepath: str) -> List[AnnotatedVariable]:
    """Parse ``filepath`` and return the list of annotated variables.

    Raises
    ------
    ValueError
        On unknown tags, duplicate variable names, or annotations that are
        not followed by an assignment.
    """
    with open(filepath, "r", encoding="utf-8") as f:
        source = f.read()

    tree = ast.parse(source, filename=filepath)
    annotations = find_annotation_lines(source)

    variables: List[AnnotatedVariable] = []
    seen_names: set = set()

    for line_no, raw_tag in annotations:
        tags = _parse_tags(raw_tag)
        unknown = tags - VALID_TAGS
        if unknown:
            bad = next(iter(unknown))
            raise ValueError(
                f"Unknown Ditto tag '{bad}' at line {line_no}. "
                f"Valid tags are: {sorted(VALID_TAGS)}."
            )
        tag = _primary_tag(tags)

        assign_node = _find_assignment_after(tree, line_no)
        if assign_node is None:
            raise ValueError(
                f"Ditto annotation '{tag}' at line {line_no} is not followed "
                f"by an assignment or pyro.sample(...) expression."
            )

        if isinstance(assign_node, ast.Expr):
            # Bare ``pyro.sample(...)`` expression statement. Use the Pyro
            # site name (the string-literal first arg) as the variable name.
            call_node = assign_node.value
            name = _extract_sample_site_name(call_node, assign_node.lineno)
            raw_expression = ast.unparse(call_node)
            is_sample_call = True
        else:
            name = _extract_target_name(assign_node)
            raw_expression = ast.unparse(assign_node.value)
            is_sample_call = _is_pyro_sample_call_ast(assign_node.value)

        if name in seen_names:
            raise ValueError(
                f"Duplicate Ditto-annotated variable name '{name}' "
                f"(line {assign_node.lineno})."
            )
        seen_names.add(name)

        variables.append(
            AnnotatedVariable(
                name=name,
                tag=tag,
                line_number=line_no,
                raw_expression=raw_expression,
                is_sample_call=is_sample_call,
                tags=tags,
            )
        )

    return variables
