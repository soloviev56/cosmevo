"""Проверки модуля cosmevo.recombination."""

import math

import numpy as np

from cosmevo import recombination as rec


def test_constants():
    assert abs(rec.B_H / 1.602176634e-12 - 13.598) < 1e-3          # эВ, с приведённой массой
    assert abs(rec.LAMBDA_LYA * 1e8 - 1215.68) < 0.01              # Å
    assert abs(rec.alpha_B(1e4, fudge=1.0) - 2.59e-13) < 0.02e-13  # случай B при 10⁴ К (Остерброк: 2,59·10⁻¹³)


def test_saha_limits():
    assert rec.saha_xe(3000) > 0.9999
    assert rec.saha_xe(800) < 1e-6
    assert rec.saha_xe(10) == 0.0                                  # e^{−B/kT} обнуляется без деления на нуль


def test_peebles_detailed_balance():
    """При C → 1 и T_m = T правая часть обращается в нуль на решении Саха."""
    z = 1500.0
    x = rec.saha_xe(z)
    dx, _ = rec._rhs(z, [x, rec.T0 * (1 + z)], rec.N_H0)
    scale = rec.alpha_B(rec.T0 * (1 + z)) * rec.N_H0 * (1 + z)**3 * x**2 / ((1 + z) * rec.hubble(z))
    assert abs(dx) < 1e-10 * scale


def test_recombination_history():
    x_e, T_m = rec.solve_recombination()
    assert 1250 < next(z for z in np.arange(1700, 900, -1.0) if x_e(z) < 0.5) < 1300
    assert 1.5e-4 < x_e(0) < 2.5e-4                                # остаточная ионизация
    assert abs(T_m(500) / (rec.T0 * 501) - 1) < 0.02
    zg = np.linspace(0, 1700, 17001)
    tau, _ = rec.optical_depth(zg, x_e)
    assert abs(np.interp(1.0, tau, zg) - 1089.8) < 5                 # z* по Planck 2018
