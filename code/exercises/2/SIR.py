import numpy as np
import matplotlib.pyplot as plt
import math

def gauss_pdf(x, mu, sigma):
    return (1 / (np.sqrt(2 * np.pi) * sigma)
        * np.exp(-0.5 * ((x - mu)**2 / sigma**2)))

def p(x):
    return (0.3 * gauss_pdf(x, 2.0, 1.0)
          + 0.4 * gauss_pdf(x, 5.0, 2.0)
          + 0.3 * gauss_pdf(x, 9.0, 1.0))

def q_uniform(x):
    return np.where((x >= 0) & (x <= 15), 1/15, 1e-12)

def q_normal(x, mu=5, sigma=4):
    return gauss_pdf(x, mu, sigma)

def sir(k, proposal, q_func, N_mult=1):
    N = k * N_mult
    samples = proposal(N)
    weights = p(samples) / q_func(samples)
    weights /= np.sum(weights)
    cdf = np.cumsum(weights)
    r = np.random.rand(k)
    idx = np.searchsorted(cdf, r)
    idx = np.clip(idx, 0, N - 1)        # guard against rounding pushing idx out of range
    return samples[idx]

x = np.linspace(-2, 16, 300)
ks = [20, 100, 1000]

fig, axes = plt.subplots(2, 3, figsize=(15, 8))
for row, (name, proposal, q_func) in enumerate([
        ("Uniform[0,15]", lambda k: np.random.uniform(0, 15, k), q_uniform),
        ("Normal(5,4)",   lambda k: np.random.normal(5, 4, k),   q_normal)]):
    for col, k in enumerate(ks):
        new_samples = sir(k, proposal, q_func, 1)
        ax = axes[row, col]
        ax.hist(new_samples, bins='fd', density=True, alpha=0.6, label="SIR samples")
        ax.plot(x, p(x), 'r', label="p(x)")
        ax.set_title(f"{name}, k={k}")
        ax.legend(fontsize=8)

plt.tight_layout()
plt.show()