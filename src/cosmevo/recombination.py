"""Рекомбинация водорода: уравнение Саха и модель трёх уровней Пиблса.

Учебная модель в духе Пиблса (1968) с поправочным множителем RECFAST
(Сигер, Саселов, Скотт, 1999):

- только водород (гелий к z ≈ 1700 уже нейтрален и учитывается лишь в
  числе частиц газа); ионизация из основного состояния прямыми
  переходами исключена (случай B);
- коэффициент рекомбинации случая B — аппроксимация Пекиньо и др. (1991)
  с множителем F = 1,14, которым RECFAST имитирует многоуровневый атом;
- выход из n = 2 — двухфотонный распад 2s → 1s и красное смещение
  фотонов Lyα (вероятность выхода Соболева);
- температура газа T_m следует за излучением через комптоновское
  рассеяние и остывает адиабатически после отрыва.

Расширение — `astropy.cosmology.Planck18`. Величины — в СГС, температура
в кельвинах.
"""

import math
from functools import lru_cache

import astropy.constants as const
import astropy.units as u
import numpy as np
from astropy.cosmology import Planck18
from scipy.integrate import solve_ivp

Y_P = 0.245                                     # массовая доля гелия (Planck 2018, BBN)
K_B = const.k_B.cgs.value
M_E = const.m_e.cgs.value
C = const.c.cgs.value
SIGMA_T = const.sigma_T.cgs.value
A_RAD = (4 * const.sigma_sb / const.c).cgs.value
# энергия связи водорода с приведённой массой, эрг
B_H = (const.Ryd * const.h * const.c / (1 + const.m_e / const.m_p)).to(u.erg).value
E_LYA = 0.75 * B_H                              # энергия перехода 2 → 1
LAMBDA_LYA = (const.h * const.c).cgs.value / E_LYA   # длина волны Lyα, см
LAMBDA_2S1S = 8.227                             # с⁻¹, двухфотонный распад 2s → 1s
F_RECFAST = 1.14
T0 = Planck18.Tcmb0.value
N_H0 = ((1 - Y_P) * Planck18.Ob0 * Planck18.critical_density0 / const.m_p).cgs.value   # см⁻³
F_HE = Y_P / (4 * (1 - Y_P))                    # n_He / n_H


def hubble(z):
    """H(z), с⁻¹."""
    return Planck18.H(z).to(1 / u.s).value


def alpha_B(T, fudge=F_RECFAST):
    """Коэффициент рекомбинации случая B, см³/с (Пекиньо и др. 1991)."""
    t = T / 1e4
    return fudge * 1e-13 * 4.309 * t**-0.6166 / (1 + 0.6703 * t**0.5300)


def saha_factor(T):
    """(2π m_e kT/h²)^{3/2}, см⁻³."""
    return (2 * math.pi * M_E * K_B * T / (2 * math.pi * const.hbar.cgs.value)**2)**1.5


def saha_xe(z, n_H0=N_H0):
    """Равновесная доля ионизации водорода x_e = n_e/n_H по Саха."""
    T, n_H = T0 * (1 + z), n_H0 * (1 + z)**3
    S = saha_factor(T) * math.exp(-B_H / (K_B * T)) / n_H        # x²/(1−x) = S
    return 2 / (1 + math.sqrt(1 + 4 / S)) if S > 0 else 0.0


def peebles_C(z, x_e, T, n_H0=N_H0):
    """Множитель Пиблса: вероятность, что атом в n = 2 дойдёт до 1s, а не ионизуется."""
    n1s = n_H0 * (1 + z)**3 * (1 - x_e)
    beta = alpha_B(T) * saha_factor(T) * math.exp(-B_H / (4 * K_B * T))
    lya = 8 * math.pi * hubble(z) / (LAMBDA_LYA**3 * n1s)          # выход Lyα
    return (LAMBDA_2S1S + lya) / (LAMBDA_2S1S + lya + beta)


def _rhs(z, y, n_H0):
    x_e, T_m = y
    T, n_H, H = T0 * (1 + z), n_H0 * (1 + z)**3, hubble(z)
    # рекомбинация при T_m, фотоионизация из n = 2 — при температуре излучения T
    beta = alpha_B(T) * saha_factor(T) * math.exp(-B_H / (4 * K_B * T))
    dx_dt = -peebles_C(z, x_e, T, n_H0) * (alpha_B(T_m) * n_H * x_e**2
                                     - beta * (1 - x_e) * math.exp(-E_LYA / (K_B * T)))
    compton = 8 * SIGMA_T * A_RAD * T**4 / (3 * M_E * C) * x_e / (1 + F_HE + x_e)
    dT_dt = -2 * H * T_m + compton * (T - T_m)
    dt_dz = -1 / ((1 + z) * H)
    return [dx_dt * dt_dz, dT_dt * dt_dz]


@lru_cache(maxsize=4)
def solve_recombination(z_start=1700.0, z_end=0.0, baryon_factor=1.0):
    """Решить уравнения Пиблса и для T_m от z_start (по Саха) до z_end.

    baryon_factor умножает плотность барионов (расширение не меняется).
    Возвращает функции x_e(z) и T_m(z) (интерполяция решения Радо).
    """
    n_H0 = N_H0 * baryon_factor
    sol = solve_ivp(_rhs, [z_start, z_end], [saha_xe(z_start, n_H0), T0 * (1 + z_start)], args=(n_H0,),
                    method="Radau", rtol=1e-8, atol=[1e-12, 1e-8], dense_output=True)
    if not sol.success:
        raise RuntimeError(sol.message)
    return (lambda z: sol.sol(z)[0]), (lambda z: sol.sol(z)[1])


def thomson_rate(z, x_e):
    """n_e σ_T c, с⁻¹."""
    return N_H0 * (1 + z)**3 * x_e * SIGMA_T * C


def optical_depth(z_grid, x_e):
    """Оптическая толщина τ(z) от наблюдателя до z (массив z по возрастанию)."""
    z_grid = np.asarray(z_grid)
    dtau = np.array([thomson_rate(z, x_e(z)) / ((1 + z) * hubble(z)) for z in z_grid])
    tau = np.concatenate([[0.0], np.cumsum(0.5 * (dtau[1:] + dtau[:-1]) * np.diff(z_grid))])
    return tau, dtau
