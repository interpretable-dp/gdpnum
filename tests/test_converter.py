import numpy as np
import pytest

from scipy import optimize, stats

from gdpnum import PLDConverter

from dp_accounting.pld.privacy_loss_distribution import (
    from_gaussian_mechanism,
    from_laplace_mechanism,
    from_randomized_response,
)

TOL = 1e-4


@pytest.mark.parametrize("mu", [0.5, 1.0, 2.0, 2.5])
def test_gaussian_converter_has_no_regret(mu):
    pld = from_gaussian_mechanism(1 / mu)
    converter = PLDConverter(pld)

    assert mu == pytest.approx(converter.get_mu(), abs=TOL)
    mu_prime, regret = converter.get_mu_and_regret()
    assert mu == pytest.approx(mu_prime, abs=TOL)
    assert regret < TOL


@pytest.mark.parametrize("b", [1.0, 1.25, 1.5, 2])
def test_laplace_has_nonzero_regret(b):
    pld = from_laplace_mechanism(b)
    converter = PLDConverter(pld)
    mu, regret = converter.get_mu_and_regret()
    assert regret > 0.01


@pytest.mark.parametrize("eps", [0.75, 1.0, 1.5, 2.0])
def test_rr_has_nonzero_regret(eps):
    p = 2 / (1 + np.exp(eps))
    pld = from_randomized_response(p, num_buckets=2)
    converter = PLDConverter(pld)
    mu, regret = converter.get_mu_and_regret()

    # Proposition 6.1 in the paper.
    assert mu == pytest.approx(-2 * stats.norm.ppf(1 / (np.exp(eps) + 1)))
    assert regret > 0.01


@pytest.mark.parametrize("mu", [0.5, 1.0, 2.0, 2.5])
def test_get_beta(mu):
    mu = 1.0
    pld = from_gaussian_mechanism(1 / mu)
    converter = PLDConverter(pld)

    # Default beta values.
    beta = converter.get_beta()
    assert 1 - 2 * beta[-1] == pytest.approx(converter.get_advantage())
    assert len(beta) == 8

    # Custom beta values.
    k = 10
    alpha = np.linspace(0, 1, k)
    beta = converter.get_beta(alpha)
    assert np.all(1 - alpha - converter.get_advantage() <= beta)
    assert np.all(beta <= 1 - alpha)
    assert beta[0] == pytest.approx(1.0)
    assert beta[-1] == pytest.approx(0.0)
    for i in range(1, k - 1):
        assert beta[i] != 0


def _gdp_epsilon(mu, delta):
    """Smallest epsilon such that mu-GDP implies (epsilon, delta)-DP."""

    def delta_of(eps):
        return stats.norm.cdf(-eps / mu + mu / 2) - np.exp(eps) * stats.norm.cdf(
            -eps / mu - mu / 2
        )

    return optimize.brentq(lambda eps: delta_of(eps) - delta, 0, 100)


def test_large_scale_dpsgd_gaussian_matches_pld():
    delta = 1e-10
    err = delta / 100
    tolerance = 1e-3

    # Regime of DP-SGD on a large dataset (tiny sampling rate, many steps).
    pld = from_gaussian_mechanism(
        standard_deviation=0.5,
        sampling_prob=4e-5,
        value_discretization_interval=tolerance,
    ).self_compose(100_000)
    converter = PLDConverter(pld)
    alphas, betas = converter.alphas, converter.betas

    assert np.all(betas <= 1 - alphas)

    # Direct epsilon should be close to the epsilon we get from the curve.
    eps_pld = pld.get_epsilon_for_delta(delta)
    with np.errstate(divide="ignore", invalid="ignore"):  # suppress warnings.
        eps_curve = np.nanmax(np.log((1 - betas - delta) / alphas))
    assert eps_pld == pytest.approx(eps_curve, rel=tolerance)

    # The epsilon obtained from GDP must be valid (not below the direct one),
    # and not too loose.
    eps_gdp = _gdp_epsilon(converter.get_mu(err=err), delta)
    assert eps_pld <= eps_gdp <= eps_pld + 0.5
