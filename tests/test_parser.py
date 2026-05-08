"""Tests for :mod:`ditto.parser`."""
from __future__ import annotations

import os
import textwrap

import pytest

from ditto.parser import find_annotation_lines, parse_file

FIXTURES = os.path.join(os.path.dirname(__file__), "fixtures")


def test_parse_simple_model_returns_two_variables():
    variables = parse_file(os.path.join(FIXTURES, "simple_model.py"))
    names = [v.name for v in variables]
    assert names == ["mu", "x"]
    tags = {v.name: v.tag for v in variables}
    assert tags == {"mu": "latent", "x": "observed"}


def test_parse_linear_regression_has_three_annotated_variables():
    variables = parse_file(os.path.join(FIXTURES, "linear_regression.py"))
    names = [v.name for v in variables]
    assert names == ["slope", "intercept", "y"]
    tags = {v.name: v.tag for v in variables}
    assert tags == {"slope": "latent", "intercept": "latent", "y": "observed"}


def test_blank_lines_between_comment_and_assignment(tmp_path):
    src = textwrap.dedent(
        """
        import pyro.distributions as dist

        # !Ditto: prior


        a = dist.Normal(0., 1.)
        """
    )
    path = tmp_path / "blank.py"
    path.write_text(src)
    variables = parse_file(str(path))
    assert len(variables) == 1
    assert variables[0].name == "a"
    assert variables[0].tag == "prior"


def test_unknown_tag_raises_value_error(tmp_path):
    src = textwrap.dedent(
        """
        # !Ditto: bogus
        x = 1
        """
    )
    path = tmp_path / "bad.py"
    path.write_text(src)
    with pytest.raises(ValueError, match="Unknown Ditto tag"):
        parse_file(str(path))


def test_approx_tag_no_longer_supported(tmp_path):
    """The legacy ``approx`` tag has been replaced by ``latent``."""
    src = textwrap.dedent(
        """
        # !Ditto: approx
        g = 1
        """
    )
    path = tmp_path / "old_tag.py"
    path.write_text(src)
    with pytest.raises(ValueError, match="Unknown Ditto tag"):
        parse_file(str(path))


def test_multiple_annotations_in_one_file(tmp_path):
    src = textwrap.dedent(
        """
        import pyro.distributions as dist

        # !Ditto: prior
        a = dist.Normal(0., 1.)

        # !Ditto: prior
        b = dist.Normal(0., 1.)

        # !Ditto: observed
        c = a + b
        """
    )
    path = tmp_path / "multi.py"
    path.write_text(src)
    variables = parse_file(str(path))
    assert [v.name for v in variables] == ["a", "b", "c"]
    assert [v.tag for v in variables] == ["prior", "prior", "observed"]


def test_find_annotation_lines_locates_correct_line_numbers():
    src = "import x\n# !Ditto: prior\na = 1\n\n# !Ditto: observed\nb = 2\n"
    annotations = find_annotation_lines(src)
    assert annotations == [(2, "prior"), (5, "observed")]


def test_duplicate_names_raise(tmp_path):
    src = textwrap.dedent(
        """
        # !Ditto: prior
        a = 1

        # !Ditto: prior
        a = 2
        """
    )
    path = tmp_path / "dup.py"
    path.write_text(src)
    with pytest.raises(ValueError, match="Duplicate"):
        parse_file(str(path))


def test_bare_pyro_sample_expression_is_annotatable(tmp_path):
    """A ``# !Ditto: observed`` above a bare ``pyro.sample(...)`` call should
    bind the annotation to that statement, using the Pyro site name."""
    src = textwrap.dedent(
        """
        import pyro
        import pyro.distributions as dist
        import torch

        def model():
            data = torch.tensor([0.0, 1.0])
            # !Ditto: observed
            pyro.sample("obs", dist.Normal(0., 1.), obs=data)
        """
    )
    path = tmp_path / "bare_sample.py"
    path.write_text(src)
    variables = parse_file(str(path))
    assert len(variables) == 1
    var = variables[0]
    assert var.name == "obs"
    assert var.tag == "observed"
    assert var.is_sample_call is True


def test_pyro_sample_with_non_string_first_arg_raises(tmp_path):
    """A non-string first arg to ``pyro.sample`` must raise ``ValueError``."""
    src = textwrap.dedent(
        """
        import pyro
        import pyro.distributions as dist

        site = "obs"
        # !Ditto: observed
        pyro.sample(site, dist.Normal(0., 1.))
        """
    )
    path = tmp_path / "non_string.py"
    path.write_text(src)
    with pytest.raises(ValueError, match="string literal"):
        parse_file(str(path))


def test_multiple_latent_variables_allowed(tmp_path):
    """Unlike the old ``approx`` rule, multiple ``latent`` vars are fine."""
    src = textwrap.dedent(
        """
        import pyro.distributions as dist

        # !Ditto: latent
        a = dist.Normal(0., 1.)

        # !Ditto: latent
        b = dist.Normal(0., 1.)
        """
    )
    path = tmp_path / "multi_latent.py"
    path.write_text(src)
    variables = parse_file(str(path))
    assert [v.name for v in variables] == ["a", "b"]
    assert [v.tag for v in variables] == ["latent", "latent"]
