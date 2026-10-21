import matplotlib.pyplot as plt
import numpy as np


def plot_rnd_walk(drift=0.1, mean=0, variance=1):
    if variance < 0:
        raise ValueError("variance must be non-negative")
    noise = np.random.default_rng().normal(loc=mean, scale=np.sqrt(variance), size=999)
    random_walk = np.concatenate(([0.0], np.cumsum(drift + noise)))

    plt.figure()
    plt.plot(random_walk)
    plt.show()