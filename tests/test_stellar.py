"""Проверки модуля cosmevo.stellar: аналитические политропы и табличные
коэффициенты (Чандрасекар, 1939, табл. 4)."""

import math

import numpy as np
from scipy.integrate import trapezoid

from cosmevo.stellar import lane_emden


def close(a, b, rel=1e-5):
    return math.isclose(a, b, rel_tol=rel)


def test_analytic_polytropes():
    p0 = lane_emden(0)                       # θ = 1 − ξ²/6
    assert close(p0.xi1, math.sqrt(6)) and close(p0.central_concentration, 1.0)
    p1 = lane_emden(1)                       # θ = sin ξ / ξ
    assert close(p1.xi1, math.pi) and close(p1.omega, math.pi)
    xi = np.linspace(0.1, 3.0, 7)
    assert np.allclose(p1.theta(xi), np.sin(xi) / xi, atol=1e-8)


def test_tabulated_coefficients():
    p15, p3 = lane_emden(1.5), lane_emden(3)
    assert close(p15.xi1, 3.65375) and close(p15.omega, 2.71406)
    assert close(p3.xi1, 6.89685) and close(p3.omega, 2.01824)
    assert close(p3.central_concentration, 54.1825)
    assert close(p3.W, 11.05, rel=1e-3)


def test_potential_energy_formula():
    # Ω = −∫ G m dm / r  против  −3/(5−n) · G M²/R  (G = M = R = 1)
    for n in (1, 1.5, 3):
        x, rho, _, m = lane_emden(n).profiles(4001)
        dm = np.gradient(m, x)
        integrand = np.where(x > 0, m * dm / np.maximum(x, 1e-12), 0.0)
        omega = -trapezoid(integrand, x)
        assert close(omega, -3 / (5 - n), rel=1e-3)
