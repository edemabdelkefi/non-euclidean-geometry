from pathlib import Path
import sys
import unittest
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from geometry_lab.spd import (
    distance, geodesic, exp_map, log_map, inner_product, parallel_transport,
    log_euclidean_mean, frechet_mean, matrix_log,
)
from geometry_lab.research import minimum_variance, portfolio_step, quasi_likelihood


class ManifoldTests(unittest.TestCase):
    def setUp(self):
        self.a = np.array([[2., .4], [.4, 1.]])
        self.b = np.array([[1., -.3], [-.3, 3.]])
        self.c = np.array([[1.2, .5], [.2, 1.5]])

    def test_congruence_invariance_of_distance_and_geodesic(self):
        ca, cb = self.c @ self.a @ self.c.T, self.c @ self.b @ self.c.T
        self.assertAlmostEqual(distance(self.a, self.b), distance(ca, cb), places=12)
        np.testing.assert_allclose(geodesic(ca, cb, .3), self.c @ geodesic(self.a, self.b, .3) @ self.c.T, atol=1e-12)

    def test_geodesic_constant_speed_and_determinant(self):
        mid = geodesic(self.a, self.b, .4)
        self.assertAlmostEqual(distance(self.a, mid), .4*distance(self.a, self.b), places=12)
        self.assertAlmostEqual(np.linalg.det(mid), np.linalg.det(self.a)**.6*np.linalg.det(self.b)**.4, places=12)
        np.testing.assert_allclose(exp_map(self.a, log_map(self.a, self.b)), self.b, atol=1e-12)

    def test_parallel_transport_preserves_metric(self):
        first = np.array([[.2, -.1], [-.1, .4]])
        second = np.array([[1., .3], [.3, -.2]])
        transported_first = parallel_transport(self.a, self.b, first)
        transported_second = parallel_transport(self.a, self.b, second)
        self.assertAlmostEqual(inner_product(self.a, first, second), inner_product(self.b, transported_first, transported_second), places=12)

    def test_diagonal_means_are_geometric(self):
        matrices = np.array([np.diag([1., 4.]), np.diag([9., 1.])])
        expected = np.diag([3., 2.])
        mean, diag = frechet_mean(matrices)
        np.testing.assert_allclose(mean, expected, atol=1e-9)
        np.testing.assert_allclose(log_euclidean_mean(matrices), expected, atol=1e-12)
        self.assertLessEqual(diag['gradient_norm'], 1e-9)

    def test_noncommuting_mean_stationarity_and_congruence(self):
        matrices = np.array([self.a, self.b, np.array([[4., .8], [.8, 2.]])])
        mean, diag = frechet_mean(matrices)
        transformed, _ = frechet_mean(np.array([self.c @ m @ self.c.T for m in matrices]))
        np.testing.assert_allclose(transformed, self.c @ mean @ self.c.T, atol=2e-8)
        self.assertTrue(np.all(np.diff(diag['objective_history']) <= 1e-12))
        self.assertLess(np.linalg.norm(sum(log_map(mean, m) for m in matrices)/3), 1e-8)

    def test_nonpositive_and_nonsymmetric_inputs_rejected(self):
        for matrix in [np.diag([1., 0.]), np.diag([-1., 2.]), np.array([[1., 2.], [0., 1.]])]:
            with self.assertRaises(ValueError):
                matrix_log(matrix)

    def test_minimum_variance_against_diagonal_analytic_solution(self):
        diagonal = np.array([1., 4., 9.])
        analytic = (1/diagonal)/(1/diagonal).sum()
        weights = minimum_variance(np.diag(diagonal), cap=1.)
        np.testing.assert_allclose(weights, analytic, atol=1e-7)
        with self.assertRaises(ValueError):
            minimum_variance(np.eye(3), cap=.2)

    def test_self_financing_drift_and_cost_accounting(self):
        weights, returns = np.array([.4, .6]), np.array([.1, -.05])
        gross, net, drifted = portfolio_step(weights, returns, .002)
        self.assertAlmostEqual(gross, .01)
        self.assertAlmostEqual(net, .998*1.01-1)
        np.testing.assert_allclose(drifted, [.44/1.01, .57/1.01])
        self.assertAlmostEqual(drifted.sum(), 1.)
        self.assertAlmostEqual(quasi_likelihood(np.array([[1., 2.]]), np.eye(2))[0], 2.5)


if __name__ == '__main__':
    unittest.main()
