"""Равновесная термодинамика ранней Вселенной (ħ = c = k_B = 1).

Плотности числа частиц, энергии, давление и энтропия идеального газа
бозонов и фермионов с нулевым химическим потенциалом; эффективные числа
степеней свободы g_* и g_*s Стандартной модели как функции температуры
фотонов. Масса и температура — в одних единицах (обычно ГэВ).

Две версии g_*: `g_star` — идеальный газ с резким КХД-переходом (прозрачная
модель для вывода), `g_star_lattice` — таблица по решёточным расчётам
(Saikawa, Shirai 2018; файл `data/gstar_lattice.csv`, извлечён скриптом
`scripts/data/make_gstar_lattice.py`) для количественных оценок.
"""

import math
from functools import lru_cache
from pathlib import Path

import numpy as np
from scipy.integrate import quad

FERMION, BOSON = +1, -1


def _integral(f, y, stat):
    """∫₀^∞ f(x, ε) / (e^ε ± 1) dx, ε = √(x² + y²), x = p/T, y = m/T."""
    def integrand(x):
        eps = math.hypot(x, y)
        if eps > 700:
            return 0.0
        return f(x, eps) / (math.exp(eps) + stat)
    return quad(integrand, 0, max(50.0, y + 50.0), limit=200)[0]


def equilibrium(m, T, g, stat):
    """Словарь n, rho, p, s для одного сорта частиц (g — число внутренних
    состояний, stat — FERMION или BOSON; античастицы считаются отдельным
    сортом или включаются в g)."""
    y = m / T
    c = g * T**4 / (2 * math.pi**2)
    n = g * T**3 / (2 * math.pi**2) * _integral(lambda x, e: x**2, y, stat)
    rho = c * _integral(lambda x, e: x**2 * e, y, stat)
    p = c * _integral(lambda x, e: x**4 / (3 * e), y, stat)
    return {"n": n, "rho": rho, "p": p, "s": (rho + p) / T}


def g_rho(m, T, g, stat):
    """Вклад сорта в g_*: ρ = (π²/30) g_* T⁴."""
    return equilibrium(m, T, g, stat)["rho"] / (math.pi**2 / 30 * T**4)


def g_s(m, T, g, stat):
    """Вклад сорта в g_*s: s = (2π²/45) g_*s T³."""
    return equilibrium(m, T, g, stat)["s"] / (2 * math.pi**2 / 45 * T**3)


# Частицы Стандартной модели: (имя, масса в ГэВ, g с учётом античастиц, статистика).
# Массы — PDG; для кварков u, d, s взяты токовые массы (их роль мала: выше
# T_QCD они безмассовы в сравнении с T, ниже — заключены в адроны).
PHOTON = ("γ", 0.0, 2, BOSON)
ELECTRON = ("e±", 0.000511, 4, FERMION)
NEUTRINOS = ("ν, ν̄ (3 сорта)", 0.0, 6, FERMION)
LEPTONS_HEAVY = [("μ±", 0.10566, 4, FERMION), ("τ±", 1.777, 4, FERMION)]
QUARKS = [("u", 0.0022, 12, FERMION), ("d", 0.0047, 12, FERMION),
          ("s", 0.095, 12, FERMION), ("c", 1.27, 12, FERMION),
          ("b", 4.18, 12, FERMION), ("t", 172.7, 12, FERMION)]
GLUONS = ("g", 0.0, 16, BOSON)
BOSONS_EW = [("W±", 80.37, 6, BOSON), ("Z", 91.19, 3, BOSON), ("H", 125.2, 1, BOSON)]
PIONS = [("π±", 0.13957, 2, BOSON), ("π⁰", 0.13498, 1, BOSON)]

T_QCD = 0.15     # ГэВ: условная граница кварк-глюонной плазмы и адронного газа
T_NU_DEC = 0.002 # ГэВ: условное мгновенное отключение нейтрино


def plasma_species(T):
    """Частицы в равновесии с фотонами при температуре T (кроме нейтрино)."""
    species = [PHOTON, ELECTRON, *LEPTONS_HEAVY, *BOSONS_EW]
    if T > T_QCD:
        species += [*QUARKS, GLUONS]
    else:
        species += PIONS
    return species


def _gs_photon_electron(T):
    return g_s(0.0, T, 2, BOSON) + g_s(ELECTRON[1], T, 4, FERMION)


def neutrino_temperature_ratio(T):
    """T_ν/T. До отключения нейтрино (T > T_NU_DEC) равно 1; после —
    из сохранения энтропии γ + e± в сопутствующем объёме при T_ν a = const:
    (T_ν/T)³ = g_s,γe(T) / g_s,γe(T_dec)."""
    if T >= T_NU_DEC:
        return 1.0
    return (_gs_photon_electron(T) / _gs_photon_electron(T_NU_DEC)) ** (1 / 3)


def g_star(T):
    """(g_*, g_*s) Стандартной модели при температуре фотонов T (ГэВ).

    Приближения: идеальный газ; резкий переход кварк-глюонная плазма →
    газ пионов при T_QCD (в действительности — плавный кроссовер);
    мгновенное отключение нейтрино при T_NU_DEC.
    """
    gr = sum(g_rho(m, T, g, st) for _, m, g, st in plasma_species(T))
    gs = sum(g_s(m, T, g, st) for _, m, g, st in plasma_species(T))
    r = neutrino_temperature_ratio(T)
    g_nu = 7 / 8 * NEUTRINOS[2]
    return gr + g_nu * r**4, gs + g_nu * r**3


# --- g_*(T) по решёточным расчётам -------------------------------------------

GSTAR_LATTICE = Path(__file__).resolve().parents[2] / "data" / "gstar_lattice.csv"


@lru_cache(maxsize=1)
def _lattice_table():
    T, gr, gs = np.loadtxt(GSTAR_LATTICE, delimiter=",", skiprows=2,
                           usecols=(0, 1, 3), unpack=True)
    return np.log(T), gr, gs


def g_star_lattice(T):
    """(g_*, g_*s) Стандартной модели по таблице Сайкавы и Сираи (2018):
    область КХД-перехода — по решёточным расчётам уравнения состояния,
    отрыв нейтрино — с N_eff = 3,045. T в ГэВ, 2·10⁻⁶ ≤ T ≤ 10¹⁷;
    линейная интерполяция по ln T. Принимает число или массив."""
    lnT, gr, gs = _lattice_table()
    x = np.log(np.asarray(T, dtype=float))
    if np.any(x < lnT[0]) or np.any(x > lnT[-1]):
        raise ValueError("T вне диапазона таблицы 2·10⁻⁶ – 10¹⁷ ГэВ")
    return np.interp(x, lnT, gr), np.interp(x, lnT, gs)
