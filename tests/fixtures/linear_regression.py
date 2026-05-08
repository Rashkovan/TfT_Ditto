"""Bayesian linear regression fixture: y = slope * x + intercept + noise.

Two latent variables feed a single observed site. The user no longer needs
to declare a guide explicitly; Ditto constructs ``AutoNormal(model)``
automatically whenever any ``latent`` variable is annotated.
"""
import torch
import pyro
import pyro.distributions as dist


def model(x, obs=None):
    slope_val = pyro.sample("slope", dist.Normal(0., 5.))
    intercept_val = pyro.sample("intercept", dist.Normal(0., 5.))
    mean = slope_val * x + intercept_val
    with pyro.plate("data", x.shape[0]):
        return pyro.sample("y", dist.Normal(mean, 1.0), obs=obs)


def get_data():
    torch.manual_seed(0)
    x = torch.linspace(-2.0, 2.0, 20)
    true_slope, true_intercept = 1.5, -0.5
    y = true_slope * x + true_intercept + 0.1 * torch.randn_like(x)
    return (x,), {"obs": y}


# !Ditto: latent
slope = dist.Normal(0., 5.)

# !Ditto: latent
intercept = dist.Normal(0., 5.)

# !Ditto: observed
y = dist.Normal(slope * 1.0 + intercept, 1.)
