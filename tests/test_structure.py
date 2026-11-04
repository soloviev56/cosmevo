"""Проверки модуля cosmevo.structure."""

import math

import numpy as np
from scipy.integrate import quad

from cosmevo import structure as st


def test_growth_normalisation_and_limits():
    assert abs(st.growth(1.0) - 1) < 1e-12
    # в эпоху вещества D ∝ a: логарифмический наклон близок к 1 при a ~ 0,01–0,05
    assert abs(st.growth_rate(0.02) - 1) < 0.03
    # сегодня f ≈ Ω_m^0,55 (Линдер) с точностью в 1 %
    assert abs(st.growth_rate(1.0) / st.omega_m(1.0)**0.55 - 1) < 0.01
    # эффект Месароша: глубоко в эпохе излучения рост логарифмически медленный
    assert st.growth(1e-5) / st.growth(1e-6) < 2


def test_sigma8_and_tophat():
    assert abs(st.sigma_R(8.0) - st.SIGMA8) < 1e-10
    assert abs(st.tophat(1e-5) - 1) < 1e-10
    x = 2.0
    assert abs(st.tophat(x) - 3 * (math.sin(x) - x * math.cos(x)) / x**3) < 1e-14
    assert st.sigma_R(1.0) > st.sigma_R(8.0) > st.sigma_R(30.0)
    assert abs(st.sigma_R(8.0, z=1.0) / st.SIGMA8 - st.growth(0.5)) < 1e-10


def test_mass_radius():
    R = st.radius_of_mass(4 * math.pi / 3 * st.RHO_M * 8.0**3)
    assert abs(R - 8.0) < 1e-12


def test_multiplicity_normalisation():
    """Вся масса — в гало: ∫ f(ν) d ln ν = 1 для ПШ и (по выбору A) для ШТ."""
    for kind in ("ps", "st"):
        total = quad(lambda nu: st.multiplicity(nu, kind) / nu, 0, np.inf, limit=200)[0]
        assert abs(total - 1) < 2e-3, kind
