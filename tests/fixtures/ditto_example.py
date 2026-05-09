import torch
import pyro
import pyro.distributions as dist
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import Adam
import matplotlib.pyplot as plt

torch.manual_seed(42)
pyro.set_rng_seed(42)

# Ground Truth params
TRUE_STRESS = 0.8 # Ditto is very stressed

CHEATED = 0 # Ditto did not cheat

KNOWLEDGE = 0.7

N_Questions = 30
N_Students = 1

a, b, sigma_time = 60.0, 30.0, 5.0 # linear mean for TimeTaken w/ sigma for variance
rho = 0.95  # P(Correct | Cheated)

# Generate Data
time_taken = dist.Normal(
    a - b * TRUE_STRESS, sigma_time
).sample()

p_correct = (
    (1 - CHEATED) * (0.25 + 0.75 * KNOWLEDGE)
    + CHEATED * rho
)

correct = dist.Binomial(
    total_count=N_Questions,
    probs=p_correct
).sample()

print(f"Time taken:         {time_taken:.1f} minutes")
print(f"Questions correct:  {correct:.0f} / {N_Questions}")

# Create Model
def model(correct, time_taken, n_questions=30):

    # !Ditto: prior, latent
    stress = pyro.sample("stress", dist.Beta(2.0, 5.0))

    mu_k = torch.sigmoid(3.0 - 4.0 * stress)
    kappa_k = 5.0

    # !Ditto: prior, latent
    knowledge = pyro.sample("knowledge",
        dist.Beta(mu_k * kappa_k, (1 - mu_k) * kappa_k))

    p_cheat = torch.sigmoid(2.0 - 6.0 * knowledge)
    # !Ditto: latent
    cheated = pyro.sample("cheated", dist.Bernoulli(p_cheat))

    # !Ditto: observed
    pyro.sample("time_taken",
        dist.Normal(60.0 - 30.0 * stress, 5.0),
        obs=time_taken)

    rho = 0.95
    p_correct = (1 - cheated) * (0.25 + 0.75 * knowledge) + cheated * rho
    # !Ditto: observed
    pyro.sample("correct",
        dist.Binomial(total_count=n_questions, probs=p_correct),
        obs=correct)

def get_data():
    return (correct, time_taken), {"n_questions": N_Questions}


if __name__ == "__main__":
    def guide(correct, time_taken, n_questions=30):
        stress_a = pyro.param("stress_a",
            torch.tensor(2.0), constraint=dist.constraints.positive)
        stress_b = pyro.param("stress_b",
            torch.tensor(5.0), constraint=dist.constraints.positive)
        pyro.sample("stress", dist.Beta(stress_a, stress_b))

        knowledge_a = pyro.param("knowledge_a",
            torch.tensor(2.0), constraint=dist.constraints.positive)
        knowledge_b = pyro.param("knowledge_b",
            torch.tensor(2.0), constraint=dist.constraints.positive)
        pyro.sample("knowledge", dist.Beta(knowledge_a, knowledge_b))

        p_cheated = pyro.param("p_cheated",
            torch.tensor(0.2), constraint=dist.constraints.unit_interval)
        pyro.sample("cheated", dist.Bernoulli(p_cheated))

    pyro.clear_param_store()
    svi = SVI(model, guide, optim=Adam({"lr": 0.01}), loss=Trace_ELBO())

    for step in range(2000):
        loss = svi.step(correct, time_taken)
        if step % 500 == 0:
            print(f"[step {step:4d}] loss: {loss:.2f}")

    stress_a  = pyro.param("stress_a").item()
    stress_b  = pyro.param("stress_b").item()
    know_a    = pyro.param("knowledge_a").item()
    know_b    = pyro.param("knowledge_b").item()
    p_cheat   = pyro.param("p_cheated").item()

    print(f"\nStress    ~ Beta({stress_a:.2f}, {stress_b:.2f})")
    print(f"Knowledge ~ Beta({know_a:.2f}, {know_b:.2f})")
    print(f"P(Cheated=1) = {p_cheat:.3f}")