import unittest
from unittest.mock import patch

import numpy as np

from GPT_tools import image_charge as ic


class ParticleStub:
    def __init__(self, n=32):
        self.n = n

    def __len__(self):
        return self.n


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
        integrand = np.minimum(q, K / mass) * np.sqrt(Eexc - K)
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

    def test_default_mass_is_one(self):
        for affinity in (-0.2, 0, 0.2):
            for fn in (ic.EcumulprobSemi, ic.dEcumulprobSemi, ic.invEcumulSemi):
                np.testing.assert_array_equal(fn(np.array([0.3, 0.7]), 1, affinity),
                                              fn(np.array([0.3, 0.7]), 1, affinity, effective_mass=1))
            self.assertEqual(ic.MTE_model_semi(1, 0, affinity),
                             ic.MTE_model_semi(1, 0, affinity, effective_mass=1))
            self.assertEqual(ic.QE_model_semi(1, 0.1, 0, affinity),
                             ic.QE_model_semi(1, 0.1, 0, affinity, effective_mass=1))
            a = ic.MakeSemiconductorEnergyDist(ParticleStub(), 1, affinity, np.random.default_rng(9))
            b = ic.MakeSemiconductorEnergyDist(ParticleStub(), 1, affinity, np.random.default_rng(9), effective_mass=1)
            np.testing.assert_array_equal(a.px, b.px)
            np.testing.assert_array_equal(a.pz, b.pz)

    def test_zero_affinity_and_nea_limits(self):
        for mass in (0.05, 0.2, 1, 2):
            self.assertAlmostEqual(ic.MTE_model_semi(0.3, 0, 0, mass), 2 * min(mass, 1) * 0.3 / 7)
            if mass <= 1:
                self.assertAlmostEqual(ic.MTE_model_semi(0.3, 0, -0.1, mass), 2 * mass * 0.3 / 7)
        # When K/r is the active branch everywhere, W is K*sqrt(Eexc-K)/r.
        self.assertAlmostEqual(ic.MTE_model_semi(1.1, 0, 1, 0.5), 2 * 0.1 / 7)

    def test_cdf_and_pdf_across_both_crossing_directions(self):
        for affinity, mass in ((-0.2, 2), (0.2, 0.2), (-0.2, 1), (0, 1)):
            energy = 1 - affinity
            lo = max(0, -affinity)
            grid = np.linspace(lo, energy, 41)
            norm = reference_integral(energy, affinity, mass, energy)
            expected = [reference_integral(energy, affinity, mass, k) / norm for k in grid]
            np.testing.assert_allclose(ic.EcumulprobSemi(grid, energy, affinity, mass), expected, atol=1e-8)
            interior = grid[1:-1]
            # At a branch crossing the PDF has a slope kink, so the symmetric
            # CDF difference has O(eps), rather than O(eps**2), truncation error.
            eps = 1e-7
            derivative = (ic.EcumulprobSemi(interior + eps, energy, affinity, mass)
                          - ic.EcumulprobSemi(interior - eps, energy, affinity, mass)) / (2 * eps)
            np.testing.assert_allclose(ic.dEcumulprobSemi(interior, energy, affinity, mass), derivative, atol=1e-7)
            np.testing.assert_array_equal(ic.EcumulprobSemi([lo - 1, energy + 1], energy, affinity, mass), [0, 1])
            np.testing.assert_array_equal(ic.dEcumulprobSemi([lo - 1, energy + 1], energy, affinity, mass), [0, 0])

    def test_broadcast_and_empty_populations(self):
        h = np.array([[0.3], [0.5]])
        v = np.array([0.0, 0.1, 0.2])
        actual = ic.MTE_model_semi(h, v, -0.1, 0.2)
        transmission = ic.QE_model_semi(h, v, 0, -0.1, 0.2)
        for i in range(2):
            for j in range(3):
                self.assertEqual(actual[i, j], ic.MTE_model_semi(h[i, 0], v[j], -0.1, 0.2))
                self.assertEqual(transmission[i, j], ic.QE_model_semi(h[i, 0], v[j], 0, -0.1, 0.2))
        self.assertEqual(ic.QE_model_semi(0.3, 0.4, 0, 0), 0)
        self.assertTrue(np.isnan(ic.MTE_model_semi(0.3, 0.4, 0)))
        self.assertTrue(np.isnan(ic.QE_model_semi(0.3, 0.4, 0.4, 0)))

    def test_invalid_mass(self):
        for mass in (0, -1, np.nan, np.inf, None):
            for fn, args in ((ic.MTE_model_semi, (1, 0, 0)),
                             (ic.QE_model_semi, (1, 0, 0, 0)),
                             (ic.invEcumulSemi, (0, 1, 0)),
                             (ic.EcumulprobSemi, (0.5, 1, 0)),
                             (ic.dEcumulprobSemi, (0.5, 1, 0))):
                with self.assertRaises(ValueError):
                    fn(*args, effective_mass=mass)

    def test_surface_filtering_and_survivor_wrapper(self):
        n, mass = 100_000, 0.2
        surface = ic.MakeSemiconductorEnergyDist(ParticleStub(n), 0.4, -0.1,
                                                np.random.default_rng(43), effective_mass=mass)
        scale = 1010.93912**2
        transverse = (surface.px**2 + surface.py**2) / scale
        survives = surface.pz**2 / scale > 0.2
        expected_mte = ic.MTE_model_semi(0.3, 0.2, -0.1, mass)
        expected_t = ic.QE_model_semi(0.3, 0.2, 0, -0.1, mass)
        self.assertAlmostEqual(survives.mean(), expected_t, delta=0.004)
        self.assertAlmostEqual(transverse[survives].mean(), expected_mte, delta=0.0003)
        with patch.object(ic, 'getSemiconductorEexc', return_value=(0.4, 0.2, -0.1)), \
                patch.object(ic, 'get_cathode_particlegroup', side_effect=lambda *a, **k: ParticleStub(n), create=True):
            pg = ic.MakeSemiconductorParticleGroup({'effective_mass': mass}, only_survivors=True,
                                                  verbose=False, rng=np.random.default_rng(44))
            default_pg = ic.MakeSemiconductorParticleGroup({}, verbose=False, rng=np.random.default_rng(45))
        self.assertTrue(np.all(pg.pz**2 / scale >= 0.2))
        self.assertAlmostEqual(np.mean((pg.px**2 + pg.py**2) / scale), expected_mte, delta=0.0003)
        expected_default = ic.MTE_model_semi(0.3, 0, -0.1, 1)
        self.assertAlmostEqual(np.mean((default_pg.px**2 + default_pg.py**2) / scale), expected_default, delta=0.0006)


if __name__ == '__main__':
    unittest.main()
