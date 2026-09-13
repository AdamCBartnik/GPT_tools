"""Standalone DOS/flux proposal checks; does not change image_charge.py.

Run with Python + NumPy. Prints illustrative MTE/transmission comparisons and
checks a proposed polynomial integral against independent quadrature and a
weighted Monte Carlo calculation in internal energy and solid angle.
"""

import numpy as np


def population_integral(delta, chi, mass, upper=None, moment=False):
    """Proposed population-flux integral in internal kinetic energy q.

    Integrates min(q, (q-chi)/mass)*sqrt(delta-q); for moment=True also
    multiplies by mean transverse energy = min(q-chi, mass*q)/2.
    Four-point Gaussian quadrature is exact for each transformed polynomial
    branch (up to floating-point roundoff). This scalar prototype deliberately
    leaves API integration and vectorized inverse-CDF sampling for later.
    """
    if not np.isfinite(mass) or mass <= 0:
        raise ValueError('mass must be finite and positive')
    lower = max(0.0, chi)
    upper = delta if upper is None else min(delta, max(lower, upper))
    if upper <= lower:
        return 0.0
    crossing = chi / (1 - mass) if mass != 1 else -1
    bounds = [lower]
    if lower < crossing < upper:
        bounds.append(crossing)
    bounds.append(upper)
    nodes, weights = np.polynomial.legendre.leggauss(4)
    total = 0.0
    for lo, hi in zip(bounds[:-1], bounds[1:]):
        t_hi, t_lo = np.sqrt(delta - lo), np.sqrt(delta - hi)
        width = (hi - lo) / (t_hi + t_lo)
        fraction = (nodes + 1) / 2
        t = t_hi - fraction * width
        increment = fraction * width * (t_hi + t)
        q = lo + increment
        k = (lo - chi) + increment
        weight = np.minimum(q, k / mass)
        integrand = t**2 * weight
        if moment:
            integrand *= mass * weight / 2
        total += np.sum(weights * width * integrand)
    return total


def reference(delta, chi, mass, model, upper=None):
    """Independent quadrature of each model, with q=lo+span*sin(theta)^2."""
    lower = max(0.0, chi)
    upper = delta if upper is None else min(delta, max(lower, upper))
    if upper <= lower:
        return 0.0, np.nan
    crossing = chi / (1 - mass) if mass != 1 else -1
    bounds = [lower]
    if lower < crossing < upper:
        bounds.append(crossing)
    bounds.append(upper)
    nodes, weights = np.polynomial.legendre.leggauss(128)
    total = numerator = 0.0
    for lo, hi in zip(bounds[:-1], bounds[1:]):
        theta = (nodes + 1) * np.pi / 4
        q = lo + (hi - lo) * np.sin(theta)**2
        k = q - chi
        jacobian = (hi - lo) * np.sin(2 * theta) * np.pi / 4
        acceptance = np.clip(k / (mass * q), 0, 1)
        root = np.sqrt(np.maximum(0.0, delta - q))
        mean_transverse = np.minimum(k, mass * q) / 2
        if model == 'legacy':
            density = np.minimum(q, k / mass) * np.sqrt(q) * root
        elif model == 'population_flux':
            density = np.minimum(q, k / mass) * root
        elif model == 'ballistic_counts':
            mu_min = np.sqrt(1 - acceptance)
            one_minus_mu = acceptance / (1 + mu_min)
            density = np.sqrt(q) * root * one_minus_mu
            mean_transverse = mass * q * one_minus_mu * (2 + mu_min) / 3
        else:
            raise ValueError(model)
        total += np.sum(weights * jacobian * density)
        numerator += np.sum(weights * jacobian * density * mean_transverse)
    return total, numerator / total


def check_polynomial_integral():
    cases = 0
    for chi in (-1.0, -0.1, 0.0, 0.1, 0.5, 0.99, 1 - 1e-8):
        for mass in (0.05, 0.2, 1.0, 2.0):
            lower = max(0.0, chi)
            for fraction in (0.01, 0.5, 1.0):
                upper = lower + fraction * (1 - lower)
                norm, mte = reference(1, chi, mass, 'population_flux', upper)
                actual = population_integral(1, chi, mass, upper)
                numerator = population_integral(1, chi, mass, upper, moment=True)
                np.testing.assert_allclose(actual, norm, rtol=2e-6, atol=0)
                np.testing.assert_allclose(numerator / actual, mte, rtol=2e-6, atol=0)
                cases += 1
    for mass in (0.05, 0.2, 1.0, 2.0):
        norm = population_integral(1, 0, mass)
        mte = population_integral(1, 0, mass, moment=True) / norm
        np.testing.assert_allclose(mte, 2 * min(mass, 1) / 7, rtol=1e-14)
    print(f'PASS: {cases} polynomial/reference comparisons and zero-affinity limits')


def check_internal_monte_carlo():
    rng = np.random.default_rng(51827)
    delta, mass, chi0, chi = 0.3, 0.2, -0.1, 0.1
    # DOS product is Beta(3/2,3/2); isotropic solid angle means uniform mu.
    q = delta * rng.beta(1.5, 1.5, 1_000_000)
    mu = rng.random(len(q))
    transverse = mass * q * (1 - mu**2)
    surface = q - chi0 >= transverse
    peak = q - chi >= transverse
    for model in ('legacy', 'population_flux', 'ballistic_counts'):
        # The legacy model's extra sqrt(q) can be represented as an extra
        # energy-dependent population multiplier, exposing its physical effect.
        weights = {'legacy': q * mu,
                   'population_flux': np.sqrt(q) * mu,
                   'ballistic_counts': np.ones_like(q)}[model]
        mte = np.sum(weights[peak] * transverse[peak]) / np.sum(weights[peak])
        transmission = np.sum(weights[peak]) / np.sum(weights[surface])
        norm, reference_mte = reference(delta, chi, mass, model)
        surface_norm, _ = reference(delta, chi0, mass, model)
        np.testing.assert_allclose(mte, reference_mte, rtol=0.006)
        np.testing.assert_allclose(transmission, norm / surface_norm, rtol=0.006)
        print(f'PASS: internal Monte Carlo {model}: MTE={1000*mte:.4f} meV, '
              f'T={transmission:.6f}')


def print_comparison():
    print('\nIllustration: delta=0.3 eV, mass=0.2, surface chi=-0.1 eV')
    print('peak_chi_eV,model,MTE_meV,surface_to_peak_transmission')
    for chi in (0.0, 0.1, 0.2, 0.29):
        for model in ('legacy', 'population_flux', 'ballistic_counts'):
            norm, mte = reference(0.3, chi, 0.2, model)
            surface_norm, _ = reference(0.3, -0.1, 0.2, model)
            print(f'{chi:.2f},{model},{1000*mte:.6f},{norm/surface_norm:.9f}')


if __name__ == '__main__':
    check_polynomial_integral()
    check_internal_monte_carlo()
    print_comparison()
