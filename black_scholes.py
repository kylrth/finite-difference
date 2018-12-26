"""Estimation of the solution to the Black-Scholes equation using the Euler-Maruyama method. Code written by Kyle Roth.

2018-12-17
"""


import numpy as np
from pandas import read_csv
from matplotlib import pyplot as plt


def euler_maruyama(t, x_0, drift, volatility):
    """Return an estimate of the solution to the Black-Scholes equation using the Euler-Maruyama method, which is
    essentially a simple finite difference method.

    Drift and volatility are assumed to return values for mu and sigma over the time steps in t.

    Parameters:
        t (ndarray): set of time steps to estimate at
        x_0 (float): initial value of the equation
        drift (callable): function returning drift constant
        volatility (callable): function returning volatility constant
    """
    del_t = t[1:] - t[:-1]
    del_wt = np.random.normal(scale=np.sqrt(del_t))

    # scale drift and volatility to the time step
    del_x = drift(del_t) * del_t + volatility(del_t) * del_wt

    # add on x_0 to start the cumulative sum at that value
    return np.cumsum(np.concatenate(([x_0], del_x)))


def estimate_mu_sigma(data, single_vals=True):
    """Return an estimate for mu (drift) and sigma (volatility) given a list of historical values."""
    n = len(data)
    diff = data[1:] - data[:-1]
    mus = diff  # samples from the random variable for drift and volatility
    mu_hat = mus.sum() / (n - 1)

    if single_vals:  # We want a single estimate, not a sample distribution.
        sigma_hat = ((diff - mu_hat) ** 2).sum() / (n - 2)
        return mu_hat, sigma_hat
    # We want a sample distribution of mus and sigmas
    sigmas = (n - 1) / (n - 2) * ((diff - mu_hat) ** 2)  # unbiased
    return mus, sigmas


class Sampler(object):
    """Object to sample parameter estimates at the scale requested."""

    def __init__(self, samples):
        """Take in the samples to be sampled from."""
        self.samples = samples

    def sample(self, shaper=1):
        """Sample from the provided samples, in the shape of the given array if present."""
        return np.random.choice(self.samples, size=np.shape(shaper))

    def __call__(self, shaper=1):
        """Allow a sampler object to be callable."""
        return self.sample(shaper=shaper)


def estimate_value(data, final_t, samples=1):
    """Using data for the value in the past, estimate the value of the stock after `final_t` more time has passed.

    Run `samples` times and return all the values.
    """
    # use the overall mu_hat and sigma_hat
    mus, sigmas = estimate_mu_sigma(data, single_vals=False)
    mu_hat = np.mean(mus)
    mu_sampler = lambda x: mu_hat
    sigma_sampler = Sampler(sigmas)

    # we don't need to estimate intermediate values
    time_diff = np.array([0, final_t])
    results = []
    for _ in range(samples):
        results.append(euler_maruyama(time_diff, data[-1], mu_sampler, sigma_sampler)[-1])  # get just the last value
    return np.array(results)


def test_black_scholes():
    """Read the data in ./stock_prices.txt, and run several iterations of Euler-Maruyama on the estimated mu and sigma.
    """
    data = np.array(read_csv('./stock_prices.txt', index_col=0, header=None)).reshape(-1)
    N = len(data)

    # parameter estimation
    mus, sigmas = estimate_mu_sigma(data, single_vals=False)  # get a sample distribution of mus and sigmas
    mu_sampler = Sampler(mus)
    sigma_sampler = Sampler(sigmas)

    estimate_days = 262  # number of days into the future
    detailed_estimates = 5  # number of fully-drawn samples to generate
    final_estimates = 1000  # number of final estimates to produce

    # plot detailed estimates
    t = np.arange(N, N + estimate_days)
    plt.plot(np.arange(N), data, label='historical')
    for i in range(1, detailed_estimates + 1):
        plt.plot(t,
                 euler_maruyama(t, data[-1], mu_sampler, sigma_sampler),
                 label='estimation {}'.format(i),
                 alpha=0.5)

    # plot final estimates
    estimates = estimate_value(data, estimate_days, samples=final_estimates)
    print('mean of projections:', estimates.mean())
    print('variance of projections:', estimates.var())
    plt.scatter(np.ones_like(estimates) * (N + estimate_days), estimates, label='estimates', alpha=0.1, c='black')

    # add source info
    alignment = {'horizontalalignment': 'center', 'verticalalignment': 'baseline'}
    plt.text(0.6, 0.02, 'github.com/kylrth/finite-difference', fontsize=7, **alignment, transform=plt.gca().transAxes)

    plt.xlabel('Days')
    plt.ylabel('Value')
    plt.title('Projected value of ACME stock (fictional)')
    plt.legend()
    plt.savefig('test_black_scholes.png')
    plt.close()


if __name__ == '__main__':
    test_black_scholes()
