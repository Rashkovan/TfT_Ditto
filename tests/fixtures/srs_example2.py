'''
IMPORTANT - We hoped Claude could make the model representable in the 
DAG form we specified but it couldn't. At least not in a mathematically 
correct manner lol. This is the file representing Claude's attempt.
'''
import pyro
import pyro.distributions as dist
import torch


def model(n_successes, n_failures):
    # Learner ability; higher = more proficient
    # !Ditto: prior, latent
    proficiency = pyro.sample("proficiency", dist.Beta(torch.tensor(2.0), torch.tensor(2.0)))

    # Rate at which successes accumulate (driven by proficiency)
    # !Ditto: latent
    lambda_minus = pyro.sample(
        "lambda_minus",
        dist.Gamma(2.0 * proficiency + 0.1, torch.tensor(1.0)),
    )

    # Rate at which failures accumulate (driven by 1 - proficiency)
    # !Ditto: latent
    lambda_plus = pyro.sample(
        "lambda_plus",
        dist.Gamma(2.0 * (1.0 - proficiency) + 0.1, torch.tensor(1.0)),
    )

    # Long-run recall probability given the learner's proficiency
    # !Ditto: prior, latent
    p_recall = pyro.sample(
        "p_recall",
        dist.Beta(5.0 * proficiency + 0.5, 5.0 * (1.0 - proficiency) + 0.5),
    )

    # Observed success count
    # !Ditto: observed
    obs_successes = pyro.sample(
        "obs_successes",
        dist.Poisson(lambda_plus),
        obs=torch.tensor(float(n_successes)),
    )

    # Observed failure count
    # !Ditto: observed
    obs_failures = pyro.sample(
        "obs_failures",
        dist.Poisson(lambda_minus),
        obs=torch.tensor(float(n_failures)),
    )

    return obs_successes, obs_failures


def guide(n_successes, n_failures):
    alpha_p = pyro.param("alpha_p", torch.tensor(2.0), constraint=dist.constraints.positive)
    beta_p = pyro.param("beta_p", torch.tensor(2.0), constraint=dist.constraints.positive)
    proficiency = pyro.sample("proficiency", dist.Beta(alpha_p, beta_p))

    alpha_lm = pyro.param("alpha_lm", torch.tensor(2.0), constraint=dist.constraints.positive)
    beta_lm = pyro.param("beta_lm", torch.tensor(1.0), constraint=dist.constraints.positive)
    pyro.sample("lambda_minus", dist.Gamma(alpha_lm, beta_lm))

    alpha_lp = pyro.param("alpha_lp", torch.tensor(2.0), constraint=dist.constraints.positive)
    beta_lp = pyro.param("beta_lp", torch.tensor(1.0), constraint=dist.constraints.positive)
    pyro.sample("lambda_plus", dist.Gamma(alpha_lp, beta_lp))

    alpha_r = pyro.param("alpha_r", torch.tensor(2.0), constraint=dist.constraints.positive)
    beta_r = pyro.param("beta_r", torch.tensor(2.0), constraint=dist.constraints.positive)
    pyro.sample("p_recall", dist.Beta(alpha_r, beta_r))


def get_data():
    return (4, 2), {}


if __name__ == "__main__":
    from pyro.infer import SVI, Trace_ELBO
    from pyro.optim import Adam

    pyro.clear_param_store()
    n_successes, n_failures = 4, 2

    svi = SVI(model, guide, Adam({"lr": 0.05}), loss=Trace_ELBO())
    for step in range(2000):
        loss = svi.step(n_successes, n_failures)
        if step % 500 == 0:
            print(f"step {step:4d} | ELBO loss: {loss:.4f}")

    print(f"proficiency mean: {pyro.param('alpha_p').item() / (pyro.param('alpha_p').item() + pyro.param('beta_p').item()):.3f}")
    print(f"p_recall mean:    {pyro.param('alpha_r').item() / (pyro.param('alpha_r').item() + pyro.param('beta_r').item()):.3f}")
