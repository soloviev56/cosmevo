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


def test_fermi_gas_limits():
    """Нерелятивистский и ультрарелятивистский пределы давления."""
    from cosmevo.stellar import M_E, fermi_energy_density, fermi_number_density, fermi_pressure
    import astropy.constants as const
    hbar, c = const.hbar.cgs.value, const.c.cgs.value
    for x in (1e-3, 1e3):
        n = fermi_number_density(x)
        nr = (3 * math.pi**2) ** (2 / 3) / 5 * hbar**2 / M_E * n ** (5 / 3)
        ur = (3 * math.pi**2) ** (1 / 3) / 4 * hbar * c * n ** (4 / 3)
        assert close(float(fermi_pressure(x)), nr if x < 1 else ur, rel=1e-3)
    eps = float(fermi_energy_density(1e-3))       # ≈ n m c²
    assert close(eps, float(fermi_number_density(1e-3)) * M_E * c**2, rel=1e-5)


def test_chandrasekhar_and_ov_limits():
    from cosmevo.stellar import neutron_star_ov, white_dwarf
    import astropy.constants as const
    M_sun = const.M_sun.cgs.value
    M, _ = white_dwarf(300.0)                      # почти предельный карлик
    assert 1.44 < M / M_sun < 1.457                # M_Ch(μ_e = 2) = 1,4563 M☉
    masses = [neutron_star_ov(x)[0] / M_sun for x in np.linspace(0.6, 1.0, 9)]
    assert close(max(masses), 0.71, rel=1e-2)      # Оппенгеймер и Волков (1939)


def test_electron_pressure_limits():
    from cosmevo.stellar import M_E, electron_pressure, fermi_pressure
    import astropy.constants as const
    k_B, hbar, c = const.k_B.cgs.value, const.hbar.cgs.value, const.c.cgs.value
    assert close(electron_pressure(1e22, 1e7), 1e22 * k_B * 1e7, rel=1e-3)   # больцмановский газ
    n_e = 1e30
    x_F = hbar * (3 * math.pi**2 * n_e) ** (1 / 3) / (M_E * c)
    assert close(electron_pressure(n_e, 1e4), fermi_pressure(x_F), rel=1e-6)  # вырожденный


def test_one_zone_model():
    from cosmevo.stellar import M_U, lane_emden, one_zone_temperature
    import astropy.constants as const
    G, k_B, M_sun = const.G.cgs.value, const.k_B.cgs.value, const.M_sun.cgs.value
    # разреженный газ: kT = μ m_u W G M^{2/3} ρ^{1/3}, 1/μ = 1/μ_i + 1/μ_e
    W = (4 * math.pi) ** (1 / 3) / (4 * lane_emden(3).omega ** (2 / 3))
    mu = 1 / (1 / 4 + 1 / 2)
    T_ideal = mu * M_U * W * G * M_sun ** (2 / 3) * 1e-3 ** (1 / 3) / k_B
    assert close(one_zone_temperature(M_sun, 1e-3, 4, 2), T_ideal, rel=1e-4)
    # ниже предела Чандрасекара (1,456 M☉ при μ_e = 2) плотное ядро холодное,
    # выше — температура растёт неограниченно
    assert math.isnan(one_zone_temperature(1.40 * M_sun, 1e10, 14, 2))
    T1, T2 = (one_zone_temperature(1.50 * M_sun, rho, 14, 2) for rho in (1e9, 1e10))
    assert T2 > T1 > 1e9


def test_one_zone_track_consistent():
    from cosmevo.stellar import one_zone_temperature, one_zone_track
    import astropy.constants as const
    M = 0.5 * const.M_sun.cgs.value
    for eta in (-5.0, 0.0, 5.0):
        rho, T = one_zone_track(M, 4, 2, eta)
        assert close(one_zone_temperature(M, rho, 4, 2), T, rel=1e-5)
