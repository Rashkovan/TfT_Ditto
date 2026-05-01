"""
Example Pyro model for Ditto.
Run Ditto, then upload this file to see live distribution curves.
Upload a modified copy (e.g. change Normal(0., 5.) to Normal(1., 2.))
to see the before/after comparison.
"""
import pyro
import pyro.distributions as dist
import torch


# !Ditto: prior
slope = dist.Normal(0., 5.)

# !Ditto: prior
intercept = dist.Normal(0., 2.)

# !Ditto: prior
noise_scale = dist.HalfNormal(1.)

# !Ditto: prior
weight = dist.Beta(2., 5.)
