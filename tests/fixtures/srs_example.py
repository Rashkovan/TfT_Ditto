
import pyro
from pyro import distributions as dist
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import Adam
import torch

class InterEventDistribution(dist.TorchDistribution):
    arg_constraints = {}
    support = dist.constraints.positive

    def __init__(self, lambda_plus, lambda_minus):
        self.lambda_plus  = lambda_plus
        self.lambda_minus = lambda_minus
        self.h     = (lambda_minus / (lambda_plus + lambda_minus)) * lambda_minus
        super().__init__(batch_shape=torch.Size([]), event_shape=torch.Size([2]))

    def log_prob(self, x):
        delta_t    = x[..., 0]
        event_type = x[..., 1]

        log_survival = (
            -delta_t
            + (1 - torch.exp(-self.h * delta_t)) / self.h
        )

        log_h       = torch.log(1 - torch.exp(-self.h * delta_t) + 1e-8)
        log_success = torch.log(self.lambda_plus).expand_as(delta_t)

        log_event = torch.where(event_type == 1, log_success, log_h)

        return log_survival + log_event
    
    def sample(self, sample_shape: torch.Size = torch.Size()) -> torch.Tensor:
        with torch.no_grad():
            t_plus   =  dist.Exponential(self.lambda_plus).sample(sample_shape)
            t_minus  = dist.Exponential(self.lambda_minus).sample(sample_shape)

            delta_t    = torch.min(t_plus, t_minus)
            event_type = (t_plus < t_minus).float()

            return torch.stack([delta_t, event_type], dim=-1)


# Create Fake Data
session_a = [
    (1, 0),
    (1, 2),
    (0, 3),
    (1, 1),
    (1, 4)
]

session_b = [
    (0, 0),
    (1, 1),
    (1, 1),
    (1, 2),
    (1, 5)
]

# Model
def model(events):
    
    # Priors
    # !Ditto: latent
    lambda_plus = pyro.sample("lambda_plus",   dist.Gamma(1.0, 1.0))
    # !Ditto: latent
    lambda_minus = pyro.sample("lambda_minus", dist.Gamma(1.0, 1.0))

    # encode data
    data = torch.tensor([[dt, et] for et, dt in events])

    with pyro.plate("observations", len(events)):
        # !Ditto: observed
        pyro.sample(
            "obs",
            InterEventDistribution(lambda_plus, lambda_minus),
            obs=data
        )

# Guide
def guide(events):
    # Variational params for lambdas
    alpha_plus = pyro.param("alpha_plus", torch.tensor(1.0), constraint=dist.constraints.positive)
    beta_plus = pyro.param("beta_plus", torch.tensor(1.0), constraint=dist.constraints.positive)
    alpha_minus = pyro.param("alpha_minus", torch.tensor(1.0), constraint=dist.constraints.positive)
    beta_minus = pyro.param("beta_minus", torch.tensor(1.0), constraint=dist.constraints.positive)

    pyro.sample("lambda_plus", dist.Gamma(alpha_plus, beta_plus))
    pyro.sample("lambda_minus", dist.Gamma(alpha_minus, beta_minus))

def get_data():                                                                                                                                               
    return (session_a,), {}

if __name__ == "__main__":
    pyro.clear_param_store()

    svi = SVI(
        model,
        guide,
        Adam({"lr": 0.01}),
        loss=Trace_ELBO()
    )

    num_steps = 2000
    losses = []

    for step in range(num_steps):
        loss = svi.step(session_a)
        losses.append(loss)
        if step % 500 ==0:
            print(f"step {step:4d} | ELBO loss: {loss:.4f}")

    # get posterior
    alpha_plus_q  = pyro.param("alpha_plus").item()
    beta_plus_q   = pyro.param("beta_plus").item()
    alpha_minus_q = pyro.param("alpha_minus").item()
    beta_minus_q  = pyro.param("beta_minus").item()

    print(f"lambda_plus  ~ Gamma({alpha_plus_q:.3f}, {beta_plus_q:.3f})")
    print(f"  mean: {alpha_plus_q / beta_plus_q:.3f}")
    print(f"lambda_minus ~ Gamma({alpha_minus_q:.3f}, {beta_minus_q:.3f})")
    print(f"  mean: {alpha_minus_q / beta_minus_q:.3f}")


    # Get Recall Probability
    def recall_probability_svi(tau, num_samples=5000):
        tau = torch.tensor(tau)

        # Sample from variational posterior
        lambda_plus_samples  = dist.Gamma(
            torch.tensor(alpha_plus_q),
            torch.tensor(beta_plus_q)
        ).sample((num_samples,))

        lambda_minus_samples = dist.Gamma(
            torch.tensor(alpha_minus_q),
            torch.tensor(beta_minus_q)
        ).sample((num_samples,))

        h = (lambda_minus_samples / (lambda_plus_samples + lambda_minus_samples)) * lambda_minus_samples

        log_survival = -tau + (1 - torch.exp(-h * tau)) / h
        S = torch.exp(log_survival)

        return {
            "mean":  S.mean().item(),
            "lower": S.quantile(0.05).item(),
            "upper": S.quantile(0.95).item()
        }

    result = recall_probability_svi(tau=1.0)
    print(f"P(recall by t=1.0): {result['mean']:.3f} [{result['lower']:.3f}, {result['upper']:.3f}]")
