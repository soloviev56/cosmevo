"""Проверки модуля cosmevo.thermo: предельные формулы и g_* Стандартной модели."""

import math

from scipy.special import zeta

from cosmevo.thermo import (BOSON, FERMION, equilibrium, g_star,
                            neutrino_temperature_ratio)


def close(a, b, rel=1e-6):
    return math.isclose(a, b, rel_tol=rel)


def test_massless_limits():
    T = 1.0
    b, f = equilibrium(0, T, 1, BOSON), equilibrium(0, T, 1, FERMION)
    assert close(b["rho"], math.pi**2 / 30)
    assert close(f["rho"], 7 / 8 * math.pi**2 / 30)
    assert close(b["n"], zeta(3) / math.pi**2)
    assert close(f["n"], 3 / 4 * zeta(3) / math.pi**2)
    assert close(b["p"], b["rho"] / 3)


def test_nonrelativistic_limit():
    m, T = 1.0, 0.02
    n = equilibrium(m, T, 1, FERMION)["n"]
    # больцмановский предел с первой релятивистской поправкой 15T/(8m)
    boltz = (m * T / (2 * math.pi)) ** 1.5 * math.exp(-m / T) * (1 + 15 * T / (8 * m))
    assert close(n, boltz, rel=2e-3)


def test_g_star_standard_model():
    gr, gs = g_star(1e4)                 # T = 10 ТэВ: всё безмассово
    assert close(gr, 106.75, rel=2e-3) and close(gs, 106.75, rel=2e-3)
    # T = 10 эВ: фотоны и холодные нейтрино. Масса электрона при отключении
    # нейтрино (m_e/T = 0,26) сдвигает T_ν/T от (4/11)^{1/3} на 0,15 %
    gr, gs = g_star(1e-8)
    assert close(neutrino_temperature_ratio(1e-8), (4 / 11) ** (1 / 3), rel=2e-3)
    assert close(gr, 3.363, rel=3e-3) and close(gs, 3.909, rel=3e-3)


def test_g_star_lattice():
    import numpy as np
    import pytest

    from cosmevo.thermo import g_star_lattice

    # низкие T: фотоны и нейтрино, N_eff ≈ 3,04 вместо 3 у мгновенного отрыва
    gr, gs = g_star_lattice(1e-5)
    n_eff = (gr - 2) / (2 * 7 / 8 * (4 / 11) ** (4 / 3))
    assert abs(n_eff - 3.045) < 0.005 and close(gs, 3.931, rel=1e-3)
    # плато перед нуклеосинтезом совпадает с идеальным газом
    assert abs(g_star_lattice(0.01)[0] - g_star(0.01)[0]) < 0.05
    # выше электрослабого масштаба взаимодействие КХД понижает g_* на ~2 %
    assert 103 < g_star_lattice(1e4)[0] < 106.75
    # кроссовер плавный: g_* монотонно растёт через область 0,1–0,3 ГэВ
    T = np.linspace(0.1, 0.3, 50)
    assert np.all(np.diff(g_star_lattice(T)[0]) > 0)
    with pytest.raises(ValueError):
        g_star_lattice(1e-7)
