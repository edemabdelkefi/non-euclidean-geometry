from pathlib import Path
import sys
import unittest
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from geometry_lab.experiment import geodesic, cross_ratio, disk_points


class GeometryTests(unittest.TestCase):
    def test_radial_distance(self):
        length, path = geodesic(np.array([0j]), np.array([.5+0j]), 512)
        self.assertAlmostEqual(length[0], np.log(3), places=10)
        np.testing.assert_allclose(path[0, [0, -1]], [0, .5], atol=1e-14)

    def test_distance_symmetry_and_quadrature_convergence(self):
        a, b = np.array([.4+.2j]), np.array([-.3+.5j])
        exact = 2*np.arctanh(abs((b-a)/(1-np.conj(a)*b)))
        coarse, _ = geodesic(a, b, 16)
        fine, path = geodesic(a, b, 512)
        reverse, _ = geodesic(b, a, 512)
        self.assertLess(abs(fine[0]-exact[0]), abs(coarse[0]-exact[0]))
        self.assertAlmostEqual(fine[0], reverse[0], places=9)
        self.assertTrue(np.all(np.abs(path)<1))

    def test_cross_ratio_under_known_homography(self):
        z = np.array([[0, 1, 2j, 3+1j]], dtype=complex)
        transformed = (2*z+1)/(z+4)
        np.testing.assert_allclose(cross_ratio(z), cross_ratio(transformed), atol=1e-13)

    def test_lorentz_interval_is_preserved(self):
        event = np.array([2., 1., -.3, .5])
        beta = .8
        gamma = 1/np.sqrt(1-beta**2)
        ct, x = gamma*(event[0]-beta*event[1]), gamma*(event[1]-beta*event[0])
        before = -event[0]**2+np.sum(event[1:]**2)
        after = -ct**2+x**2+np.sum(event[2:]**2)
        self.assertAlmostEqual(before, after, places=13)


if __name__ == '__main__':
    unittest.main()
