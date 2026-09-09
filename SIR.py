import numpy as np
import matplotlib.pyplot as plt

def gauss_pdf(x, mu, sigma):
    return (1 / (np.sqrt(2 * np.pi) * sigma)
        * np.exp(-0.5 * ((x - mu)**2 / (sigma)**2)))

k = 10000
proposal_distribution_1 = np.random.uniform(0, 15, k)
proposal_distribution_2 = np.random.normal(5, 4, k)

samples = proposal_distribution_1

def p(x):
    return 0.3 * gauss_pdf(x, 2.0, 1.0) + 0.4 * gauss_pdf(x, 5.0, 2.0) + 0.3 * gauss_pdf(x, 9.0, 1.0)


weights = p(samples)

weights = weights / np.sum(weights) 

cumulative_weights = np.cumsum(weights)

new_samples = np.zeros(k)

for i in range(k):
    r = np.random.rand()
    index = np.searchsorted(cumulative_weights, r)
    new_samples[i] = samples[index]

new_samples.sort()
samples.sort()

print("Original Samples:", samples)
print("")
print("New Samples after Resampling:", new_samples) 


x = np.linspace(0, 15, 1000)

plt.hist(new_samples, bins=30 , density=True, label="SIR samples")

plt.plot(x, p(x), label="Target p(x)")

plt.xlabel("x")
plt.ylabel("Probability density")
plt.legend()
plt.show()
