import ast
import re
from typing import List

from models import AnnotatedVariable

VALID_TAGS = {"prior", "observed", "approx"}
DITTO_COMMENT_RE = re.compile(r"#\s*!Ditto:\s*(\w+)", re.IGNORECASE)


def parse_annotated_variables(source: str) -> List[AnnotatedVariable]:
    """
    Two-pass parse:
      1. Regex scan to find lines with # !Ditto: <tag> comments and their line numbers.
      2. AST walk to find the first ast.Assign or ast.AnnAssign after each comment line.
    """
    lines = source.splitlines()

    # Pass 1: find comment annotations → {line_number (1-based): tag}
    comment_lines: dict[int, str] = {}
    for i, line in enumerate(lines, start=1):
        m = DITTO_COMMENT_RE.search(line)
        if m:
            tag = m.group(1).lower()
            if tag not in VALID_TAGS:
                raise ValueError(
                    f"Unknown Ditto tag '{tag}' on line {i}. "
                    f"Valid tags: {', '.join(sorted(VALID_TAGS))}"
                )
            comment_lines[i] = tag

    if not comment_lines:
        return []

    # Pass 2: AST walk to find the assignment node immediately after each comment
    try:
        tree = ast.parse(source)
    except SyntaxError as e:
        raise ValueError(f"Syntax error in model file: {e}") from e

    # Collect all top-level and nested assignment nodes with their line numbers
    assignments: list[tuple[int, ast.stmt]] = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            assignments.append((node.lineno, node))
    assignments.sort(key=lambda t: t[0])

    results: List[AnnotatedVariable] = []
    for comment_line, tag in sorted(comment_lines.items()):
        # Find first assignment whose line is strictly after the comment line
        target_node = None
        for lineno, node in assignments:
            if lineno > comment_line:
                target_node = node
                break

        if target_node is None:
            continue

        name, expr = _extract_name_expr(target_node)
        if name is None:
            continue

        results.append(AnnotatedVariable(name=name, tag=tag, expr=expr, line=comment_line))

    return results


def _extract_name_expr(node: ast.stmt) -> tuple[str | None, str]:
    """Return (variable_name, rhs_expr_string) or (None, '') to skip."""
    if isinstance(node, ast.AnnAssign):
        if node.value is None:
            return None, ""
        if not isinstance(node.target, ast.Name):
            return None, ""
        return node.target.id, ast.unparse(node.value)

    if isinstance(node, ast.Assign):
        # Skip augmented assignments (handled separately) and multi-target
        if len(node.targets) != 1:
            return None, ""
        target = node.targets[0]
        if not isinstance(target, ast.Name):
            return None, ""
        return target.id, ast.unparse(node.value)

    return None, ""
