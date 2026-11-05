"""Крупномасштабная структура: рост возмущений, линейный спектр, функции масс гало.

Учебная модель для статей об образовании структуры и первых звёздах:

- фактор роста D(a) растущей моды холодного вещества — численное
  решение линейного уравнения для вещества, излучения (фотоны +
  безмассовые нейтрино) и Λ, нормировка D(1) = 1;
- линейный спектр мощности P(k) ∝ k^{n_s} T²(k) с передаточной функцией
  Бардина–Бонда–Кайзера–Шалая (1986) для холодной тёмной материи
  (без барионных осцилляций), нормировка на σ₈;
- σ(M) — дисперсия в сфере, содержащей массу M среднего вещества;
- функции масс Пресса–Шехтера (1974) и Шета–Тормена (1999).

Опорная космология — `astropy.cosmology.Planck18`. Длины — в Мпк/h,
массы — в M☉/h, плотности числа гало — в (h/Мпк)³.
"""

import math
from functools import lru_cache

import numpy as np
from astropy.cosmology import Planck18
from scipy.integrate import solve_ivp

OM = Planck18.Om0
OR = Planck18.Ogamma0 * (1 + 3.046 * 7 / 8 * (4 / 11)**(4 / 3))   # фотоны + безмассовые ν
OL = 1 - OM - OR
H = Planck18.h
N_S = Planck18.meta["n"]
SIGMA8 = Planck18.meta["sigma8"]
RHO_CRIT_H2 = 2.775e11                       # ρ_c / h², M☉/Мпк³ (3H₀²/8πG при h = 1)
RHO_M = OM * RHO_CRIT_H2                     # средняя плотность вещества сегодня, (M☉/h)/(Мпк/h)³
DELTA_C = 3 / 20 * (12 * math.pi)**(2 / 3)   # порог коллапса Эйнштейна–де Ситтера, 1,686

_LNK = np.linspace(math.log(1e-5), math.log(1e4), 6000)   # сетка ln k для интегралов σ(R)


def hubble_E(a):
    """H(a)/H₀."""
    return np.sqrt(OM / a**3 + OR / a**4 + OL)


def omega_m(a):
    """Доля вещества Ω_m(a)."""
    return OM / a**3 / hubble_E(a)**2


@lru_cache(maxsize=1)
def _growth_solution():
    """Решение для D(ln a) от глубокой эпохи излучения (старт с решения Месароша)."""
    a_eq = OR / OM

    def rhs(lna, Y):
        a = math.exp(lna)
        e2 = OM / a**3 + OR / a**4 + OL
        dlnH = -(3 * OM / a**3 + 4 * OR / a**4) / (2 * e2)
        return [Y[1], -(2 + dlnH) * Y[1] + 1.5 * OM / a**3 / e2 * Y[0]]

    a0 = 1e-6
    y0 = a0 / a_eq
    sol = solve_ivp(rhs, [math.log(a0), 0.5], [1 + 1.5 * y0, 1.5 * y0],
                    rtol=1e-10, atol=1e-14, dense_output=True)
    return sol, sol.sol(0.0)[0]


def growth(a):
    """Фактор роста D(a), D(1) = 1 (a от 10⁻⁶ до e^0,5)."""
    sol, d1 = _growth_solution()
    return sol.sol(np.log(a))[0] / d1


def growth_rate(a):
    """f = d ln D / d ln a."""
    sol, _ = _growth_solution()
    y = sol.sol(np.log(a))
    return y[1] / y[0]


def transfer_bbks(k):
    """Передаточная функция BBKS для холодной тёмной материи; k в h/Мпк."""
    q = np.asarray(k, dtype=float) / (OM * H)
    return (np.log(1 + 2.34 * q) / (2.34 * q)
            * (1 + 3.89 * q + (16.1 * q)**2 + (5.46 * q)**3 + (6.71 * q)**4)**-0.25)


def tophat(x):
    """Фурье-образ сферического окна; при малых x — ряд, чтобы не терять точность."""
    x = np.asarray(x, dtype=float)
    small = x < 1e-3
    xs = np.where(small, 1.0, x)
    return np.where(small, 1 - x**2 / 10, 3 * (np.sin(xs) - xs * np.cos(xs)) / xs**3)


def _sigma2_unnorm(R):
    k = np.exp(_LNK)
    integrand = k**3 / (2 * np.pi**2) * k**N_S * transfer_bbks(k)**2 * tophat(np.outer(R, k))**2
    return np.trapezoid(integrand, _LNK, axis=-1)


@lru_cache(maxsize=1)
def _amplitude():
    return SIGMA8**2 / _sigma2_unnorm(np.array([8.0]))[0]


def power_spectrum(k, z=0.0):
    """Линейный P(k) в (Мпк/h)³; k в h/Мпк."""
    k = np.asarray(k, dtype=float)
    return _amplitude() * k**N_S * transfer_bbks(k)**2 * growth(1 / (1 + z))**2


def radius_of_mass(M):
    """Лагранжев радиус (Мпк/h) сферы с массой M (M☉/h) при средней плотности."""
    return (3 * np.asarray(M, dtype=float) / (4 * np.pi * RHO_M))**(1 / 3)


def sigma_R(R, z=0.0):
    """Линейная дисперсия σ(R) в сфере радиуса R (Мпк/h)."""
    R = np.atleast_1d(np.asarray(R, dtype=float))
    s = np.sqrt(_amplitude() * _sigma2_unnorm(R)) * growth(1 / (1 + z))
    return s if s.size > 1 else float(s[0])


def sigma_M(M, z=0.0):
    """σ(M), M в M☉/h."""
    return sigma_R(radius_of_mass(M), z)


def _f_ps(nu):
    return np.sqrt(2 / np.pi) * nu * np.exp(-nu**2 / 2)


def _f_st(nu, A=0.3222, a=0.707, p=0.3):
    an2 = a * nu**2
    return A * np.sqrt(2 * an2 / np.pi) * (1 + an2**-p) * np.exp(-an2 / 2)


def multiplicity(nu, kind="ps"):
    """f(ν): доля массы в гало на единицу ln ν (ν = δ_c/σ)."""
    nu = np.asarray(nu, dtype=float)
    return _f_ps(nu) if kind == "ps" else _f_st(nu)


def mass_function(M, z=0.0, kind="ps", delta_c=DELTA_C):
    """dn/d ln M в (h/Мпк)³ для масс M (M☉/h): Пресс–Шехтер ("ps") или Шет–Тормен ("st")."""
    M = np.atleast_1d(np.asarray(M, dtype=float))
    eps = 1e-3
    s = sigma_M(M, z)
    dlns = (np.log(sigma_M(M * (1 + eps), z)) - np.log(sigma_M(M * (1 - eps), z))) / (2 * eps)
    nu = delta_c / s
    return RHO_M / M * multiplicity(nu, kind) * np.abs(dlns)


def delta_vir_crit(z):
    """Вириальный контраст относительно критической плотности при z (Брайан, Норман, 1998)."""
    x = Planck18.Om(z) - 1
    return 18 * math.pi**2 + 82 * x - 39 * x**2


def halo_virial(M_sun, z, mu=0.59):
    """Вириальные радиус (кпк), круговая скорость (км/с) и температура (К) гало.

    M_sun — масса в M☉ (без h); mu — средняя масса частицы газа в m_p
    (0,59 — ионизованный, 1,22 — нейтральный первичный газ).
    """
    import astropy.constants as const
    M_g = M_sun * const.M_sun.cgs.value
    rho_c = Planck18.critical_density(z).cgs.value
    r = (3 * M_g / (4 * math.pi * delta_vir_crit(z) * rho_c))**(1 / 3)
    V = math.sqrt(const.G.cgs.value * M_g / r)
    T = mu * const.m_p.cgs.value * V**2 / (2 * const.k_B.cgs.value)
    return r / const.kpc.cgs.value, V / 1e5, T
