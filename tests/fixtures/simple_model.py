"""Simple two-variable Pyro model used by Ditto's parser/inference tests.

Ditto annotations live in comments above the assignments they describe. The
helper functions are defined *before* the annotated assignments so that
``load_user_module`` can execute the file end-to-end. The user no longer
needs to define a guide; Ditto auto-creates ``AutoNormal(model)`` whenever
any ``latent`` variable is found.
"""
import torch
import pyro
import pyro.distributions as dist


def model(obs=None):
    mu_val = pyro.sample("mu", dist.Normal(0., 1.))
    n = obs.shape[0] if obs is not None else 1
    with pyro.plate("data", n):
        return pyro.sample("x", dist.Normal(mu_val, 1.), obs=obs)


def get_data():
    obs = torch.tensor([0.5, 1.0, -0.3, 0.8])
    return (), {"obs": obs}


# !Ditto: latent
mu = dist.Normal(0., 1.)

# !Ditto: observed
x = dist.Normal(mu.mean, 1.)
