"""Первичный нуклеосинтез: слабые скорости n ⇄ p, расширение и сеть реакций.

Упрощённый код в духе Вагонера (1967), достаточный для учебных целей:

- слабые скорости — борновское приближение с бесконечно тяжёлыми
  нуклонами, нормированное на время жизни нейтрона; без кулоновских,
  радиационных и конечно-массовых поправок (поэтому Y_p на 0,005 ниже,
  чем у точного кода PRIMAT); нейтрино — с температурой T_ν из
  `cosmevo.thermo.neutrino_temperature_ratio`;
- темп расширения — по g_* таблицы Сайкавы–Сираи (`g_star_lattice`),
  T a g_*s^{1/3} = const;
- ядерные скорости — JINA REACLIB (`cosmevo.reaclib`), сеть из n, p, d,
  t, ³He, ⁴He, ⁶Li, ⁷Li, ⁷Be со всеми прямыми и обратными реакциями.

Температура — в МэВ, время — в секундах.
"""

import math
from functools import lru_cache

import astropy.constants as const
import astropy.units as u
import numpy as np
from scipy.integrate import quad, solve_ivp
from scipy.special import zeta

from cosmevo.nuclear import nse, nse_network
from cosmevo.reaclib import Network, load_reaclib
from cosmevo.thermo import g_star_lattice, neutrino_temperature_ratio

SPECIES = ("n", "p", "d", "t", "he3", "he4", "li6", "li7", "be7")
MASS_NUMBER = dict(zip(SPECIES, (1, 1, 2, 3, 3, 4, 6, 7, 7)))

TAU_N = 878.4                                     # с, время жизни нейтрона (PDG 2024)
M_E = (const.m_e * const.c**2).to(u.MeV).value
Q_NP = ((const.m_n - const.m_p) * const.c**2).to(u.MeV).value
M_PL = (np.sqrt(const.hbar * const.c / const.G) * const.c**2).to(u.MeV).value
HBAR = const.hbar.to(u.MeV * u.s).value           # МэВ·с
HBARC = (const.hbar * const.c).to(u.MeV * u.cm).value
M_U = const.u.cgs.value                           # г
MEV_T9 = (1 * u.MeV / const.k_B).to(u.K).value / 1e9


# --- слабые скорости ---------------------------------------------------------

def _fermi(E, T):
    x = E / T
    return 0.0 if x > 700 else 1.0 / (math.exp(x) + 1.0)


def _p(E):
    return math.sqrt(max(E * E - M_E * M_E, 0.0))


@lru_cache(maxsize=1)
def _vacuum_phase_space():
    """∫ E p (Q − E)² dE по спектру распада нейтрона, МэВ⁵."""
    return quad(lambda E: E * _p(E) * (Q_NP - E) ** 2, M_E, Q_NP)[0]


def weak_rates(T, T_nu=None, tau_n=TAU_N):
    """(λ_{n→p}, λ_{p→n}) в с⁻¹ при температуре фотонов и e± T и нейтрино
    T_nu (МэВ; по умолчанию T_nu = T). Нулевой химпотенциал нейтрино.

    n→p: n ν → p e⁻, n e⁺ → p ν̄, n → p e⁻ ν̄;
    p→n: p e⁻ → n ν, p ν̄ → n e⁺, p e⁻ ν̄ → n.
    Энергии: E_e = E_ν ± Q; общий множитель K = 1/(τ_n · фазовый объём).
    """
    T_nu = T if T_nu is None else T_nu
    K = 1.0 / (tau_n * _vacuum_phase_space())
    f, Q, top = _fermi, Q_NP, 50 * max(T, T_nu) + 10 * Q_NP
    integral = lambda g, a, b: quad(g, a, b, limit=200, epsabs=0, epsrel=1e-7)[0]
    n_to_p = (
        integral(lambda Ev: Ev**2 * f(Ev, T_nu) * (Ev + Q) * _p(Ev + Q) * (1 - f(Ev + Q, T)), 0, top)
        + integral(lambda E: E * _p(E) * f(E, T) * (E + Q) ** 2 * (1 - f(E + Q, T_nu)), M_E, top)
        + integral(lambda E: E * _p(E) * (Q - E) ** 2 * (1 - f(E, T)) * (1 - f(Q - E, T_nu)), M_E, Q))
    p_to_n = (
        integral(lambda E: E * _p(E) * f(E, T) * (E - Q) ** 2 * (1 - f(E - Q, T_nu)), Q, top)
        + integral(lambda Ev: Ev**2 * f(Ev, T_nu) * (Ev - Q) * _p(Ev - Q) * (1 - f(Ev - Q, T)),
                   Q + M_E, top)
        + integral(lambda E: E * _p(E) * (Q - E) ** 2 * f(E, T) * f(Q - E, T_nu), M_E, Q))
    return K * n_to_p, K * p_to_n


# --- расширение -------------------------------------------------------------

@lru_cache(maxsize=1)
def _gs_today():
    return float(g_star_lattice(2.0e-6)[1])


def hubble(T, dNeff=0.0):
    """H, с⁻¹, при температуре фотонов T (МэВ); dNeff — добавочные сорта
    безмассовых нейтрино с температурой T_ν (ускоряют расширение)."""
    g = float(g_star_lattice(T * 1e-3)[0]) + 7 / 4 * dNeff * neutrino_temperature_ratio(T * 1e-3) ** 4
    return math.sqrt(8 * math.pi**3 * g / 90) * T**2 / M_PL / HBAR


def dlnT_dt(T, dNeff=0.0):
    """d ln T/dt = −H/(1 + ⅓ d ln g_*s/d ln T) — из сохранения энтропии
    (добавочные нейтрино оторваны и на g_*s фотонной плазмы не влияют)."""
    h = 1.01
    gs_hi, gs_lo = (float(g_star_lattice(T * 1e-3 * x)[1]) for x in (h, 1 / h))
    return -hubble(T, dNeff) / (1 + math.log(gs_hi / gs_lo) / (3 * 2 * math.log(h)))


def baryon_density(T, eta):
    """ρ_B = m_u n_B, г/см³; n_B = η n_γ(T) g_*s(T)/g_*s(сегодня)."""
    n_gamma = 2 * zeta(3) / math.pi**2 * (T / HBARC) ** 3
    return M_U * eta * n_gamma * float(g_star_lattice(T * 1e-3)[1]) / _gs_today()


# --- сеть --------------------------------------------------------------------

@lru_cache(maxsize=1)
def network():
    """Сеть: все реакции REACLIB между SPECIES, кроме слабых (распад
    нейтрона заменён тепловыми скоростями `weak_rates`; прочие слабые
    процессы за время нуклеосинтеза пренебрежимы)."""
    rates = [r for r in load_reaclib().values()
             if not r.weak and set(r.reactants + r.products) <= set(SPECIES)]
    return Network(rates, SPECIES)


@lru_cache(maxsize=1)
def _weak_table():
    """Таблица ln λ(T) при τ_n = TAU_N; при другом τ_n все слабые скорости
    умножаются на TAU_N/τ_n (общий множитель K в `weak_rates`)."""
    lnT = np.linspace(math.log(20.0), math.log(1e-3), 300)
    lam = np.array([weak_rates(math.exp(x), math.exp(x) * neutrino_temperature_ratio(math.exp(x) * 1e-3))
                    for x in lnT])
    lam = np.maximum(lam, 1e-300)                     # λ_{p→n} → 0 при T → 0
    return lnT[::-1], np.log(lam[::-1, 0]), np.log(lam[::-1, 1])


def _weak_interp(T, tau_n):
    lnT, l_np, l_pn = _weak_table()
    x, scale = math.log(T), TAU_N / tau_n
    return scale * math.exp(np.interp(x, lnT, l_np)), scale * math.exp(np.interp(x, lnT, l_pn))


ZA = dict(zip(SPECIES, ((0, 1), (1, 1), (1, 2), (1, 3), (2, 3), (2, 4), (3, 6), (3, 7), (4, 7))))


def saha_light(T, eta, Ye):
    """Числа Y ядер сети в ядерном статистическом равновесии при
    температуре T (МэВ) и доле протонов Ye."""
    full = nse_network(A_max=7)
    pick = [int(np.flatnonzero((full["Z"] == z) & (full["A"] == a))[0]) for z, a in ZA.values()]
    net = {k: v[pick] for k, v in full.items()}
    X, _ = nse(T * MEV_T9 * 1e9, baryon_density(T, eta), Ye, net)
    return X / net["A"]


def solve_bbn(eta, T_start=10.0, T_switch=0.8, T_end=3e-3, tau_n=TAU_N, dNeff=0.0, rtol=1e-6):
    """Решить уравнения нуклеосинтеза. Возвращает словарь с массивом
    температур 'T' (МэВ, убывает), временем 'time' (с), числами частиц на
    нуклон Y[s] для каждого s из SPECIES и решениями solve_ivp ('sol').

    Этап 1 (T_start → T_switch): только n ⇄ p, ядер нет.
    Этап 2 (T_switch → T_end): полная сеть; начальный состав ядер —
    равновесие по Саха (`cosmevo.nuclear.nse`) при данном n/p.
    """
    net = network()
    idx = {s: j for j, s in enumerate(SPECIES)}

    def rhs(lnT, y, with_nuclei):
        T = math.exp(lnT)
        dt = 1.0 / dlnT_dt(T, dNeff)                       # dt/d ln T < 0
        Y = np.maximum(y[:-1], 0.0)
        lam_np, lam_pn = _weak_interp(T, tau_n)
        dY = np.zeros(len(SPECIES))
        dY[idx["n"]] = -lam_np * Y[idx["n"]] + lam_pn * Y[idx["p"]]
        dY[idx["p"]] = -dY[idx["n"]]
        if with_nuclei:
            dY += net.rhs(T * MEV_T9, Y, baryon_density(T, eta))
        return np.append(dY * dt, dt)

    def jac(lnT, y, with_nuclei):
        T = math.exp(lnT)
        dt = 1.0 / dlnT_dt(T, dNeff)
        Y = np.maximum(y[:-1], 0.0)
        lam_np, lam_pn = _weak_interp(T, tau_n)
        J = np.zeros((len(y), len(y)))
        n, p = idx["n"], idx["p"]
        J[n, n], J[n, p], J[p, n], J[p, p] = -lam_np, lam_pn, lam_np, -lam_pn
        if with_nuclei:
            J[:-1, :-1] += net.jacobian(T * MEV_T9, Y, baryon_density(T, eta))
        return J * dt

    lam_np, lam_pn = _weak_interp(T_start, tau_n)
    Y0 = np.zeros(len(SPECIES))
    Y0[idx["n"]] = lam_pn / (lam_np + lam_pn)
    Y0[idx["p"]] = 1 - Y0[idx["n"]]
    t0 = 1 / (2 * hubble(T_start, dNeff))
    s1 = solve_ivp(rhs, [math.log(T_start), math.log(T_switch)], np.append(Y0, t0),
                   args=(False,), method="LSODA", rtol=rtol, atol=1e-12, dense_output=True)
    y_switch = s1.y[:, -1].copy()
    y_switch[:-1] = saha_light(T_switch, eta, y_switch[idx["p"]])
    s2 = solve_ivp(rhs, [math.log(T_switch), math.log(T_end)], y_switch,
                   args=(True,), method="Radau", jac=jac, rtol=rtol, atol=1e-30,
                   dense_output=True)
    lnT = np.concatenate([s1.t, s2.t[1:]])
    y = np.concatenate([s1.y, s2.y[:, 1:]], axis=1)
    out = {"T": np.exp(lnT), "time": y[-1], "sol": (s1, s2)}      # "t" — тритий
    out.update({s: y[j] for j, s in enumerate(SPECIES)})
    return out


def final_abundances(res):
    """Итог: Y_p (массовая доля ⁴He), D/H, ³He/H (с тритием, который
    распадается в ³He), ⁷Li/H (с ⁷Be, который захватывает электрон)."""
    H = res["p"][-1]
    return {"Yp": 4 * res["he4"][-1],
            "D/H": res["d"][-1] / H,
            "He3/H": (res["he3"][-1] + res["t"][-1]) / H,
            "Li7/H": (res["li7"][-1] + res["be7"][-1]) / H}
