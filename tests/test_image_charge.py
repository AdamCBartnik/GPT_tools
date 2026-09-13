import unittest

import numpy as np

from GPT_tools import image_charge as ic


class ParticleStub:
    def __len__(self):
        return 32


def reference_integral(Eexc, affinity, mass, upper, moment=False):
    """Independent quadrature using K = lower + span*sin(theta)**2."""
    nodes, weights = np.polynomial.legendre.leggauss(128)
    lower = max(0.0, -affinity)
    crossing = mass * affinity / (1 - mass) if mass != 1 else -1
    bounds = [lower]
    if lower < crossing < upper:
        bounds.append(crossing)
    bounds.append(upper)
    result = 0.0
    for lo, hi in zip(bounds[:-1], bounds[1:]):
        theta = (nodes + 1) * np.pi / 4
        K = lo + (hi - lo) * np.sin(theta)**2
        q = affinity + K
        integrand = np.minimum(q, K / mass) * np.sqrt(q * (Eexc - K))
        if moment:
            integrand *= 0.5 * np.minimum(K, mass * q)
        result += np.sum(weights * integrand * (hi - lo) * np.sin(2 * theta)) * np.pi / 4
    return result


class SemiconductorRegressionTests(unittest.TestCase):
    def test_positional_rng(self):
        positional = ic.MakeSemiconductorEnergyDist(
            ParticleStub(), 1, -0.2, np.random.default_rng(42))
        keyword = ic.MakeSemiconductorEnergyDist(
            ParticleStub(), 1, -0.2, rng=np.random.default_rng(42))
        for name in ('px', 'py', 'pz'):
            np.testing.assert_array_equal(getattr(positional, name), getattr(keyword, name))

    def test_positional_solver_options(self):
        expected = ic.invEcumulSemi(0.5, 1, -0.2, ptol=1e-8, max_iter=100)
        self.assertEqual(ic.invEcumulSemi(0.5, 1, -0.2, 1e-8, 100), expected)

    def test_integrals_against_quadrature(self):
        for energy in (0.1, 0.051, 0.049, 0.01, 1e-4, 1e-5, 1e-8):
            for mass in (0.5, 1.0, 2.0, energy / 2):
                with self.subTest(energy=energy, mass=mass):
                    denom = reference_integral(energy, 1, mass, energy)
                    numer = reference_integral(energy, 1, mass, energy, moment=True)
                    np.testing.assert_allclose(ic._semi_total_weight(energy, 1, mass), denom, rtol=1e-8)
                    actual = ic._semi_total_mte_weight(energy, 1, mass) / ic._semi_total_weight(energy, 1, mass)
                    np.testing.assert_allclose(actual, numer / denom, rtol=1e-7, atol=0)
                    self.assertTrue(0 <= actual <= energy / 2)

    def test_threshold_inverse_against_independent_cdf(self):
        probabilities = np.array([0, 1e-6, 0.1, 0.5, 0.9, 1 - 1e-6, 1])
        for energy in (1e-4, 1e-5, 1e-8):
            for mass in (0.5, energy / 2):
                energies = ic.invEcumulSemi(probabilities, energy, 1, effective_mass=mass)
                norm = reference_integral(energy, 1, mass, energy)
                actual = [reference_integral(energy, 1, mass, k) / norm for k in energies]
                np.testing.assert_allclose(actual, probabilities, atol=1.1e-7, rtol=0)

    def test_keyword_mass_particle_momenta(self):
        pg = ic.MakeSemiconductorEnergyDist(
            ParticleStub(), 1e-5, 1, np.random.default_rng(42), effective_mass=0.5)
        energy = (pg.px**2 + pg.py**2 + pg.pz**2) / 1010.93912**2
        self.assertTrue(np.all(np.isfinite(energy)))
        self.assertTrue(np.all((energy >= 0) & (energy <= 1e-5)))


if __name__ == '__main__':
    unittest.main()
