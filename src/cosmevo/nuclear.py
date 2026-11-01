"""Энергии связи ядер (таблица AME2020, формула Вайцзеккера) и скорости
нерезонансных термоядерных реакций.

Таблица `data/ame2020.csv` извлечена из оценки атомных масс AME2020
(Wang M. et al., Chinese Phys. C 45, 030003, 2021) скриптом
`scripts/data/make_ame2020.py`. Энергии — в МэВ.
"""

import math

import astropy.constants as const
import astropy.units as u
import numpy as np
from scipy.integrate import quad

from cosmevo import DATA_DIR

AME2020 = DATA_DIR / "ame2020.csv"

# Названия членов формулы Вайцзеккера в порядке коэффициентов
TERMS = ("объём", "поверхность", "кулон", "асимметрия", "спаривание")


def load_ame(measured_only=True):
    """Таблица AME2020 как словарь массивов: Z, N, A, el, B (полная энергия
    связи, МэВ), BA (на нуклон, МэВ), dB, mass_excess (МэВ).

    measured_only — отбросить значения, полученные экстраполяцией («#»).
    Нейтрон и протон (A = 1) не входят: у них нет энергии связи.
    """
    with open(AME2020, encoding="utf-8") as f:
        rows = [line.rstrip("\n").split(",") for line in f if not line.startswith("#")]
    header, rows = rows[0], rows[1:]
    col = {name: np.array([r[i] for r in rows]) for i, name in enumerate(header)}
    Z, N, A = (col[k].astype(int) for k in ("Z", "N", "A"))
    keep = A > 1
    if measured_only:
        keep &= col["estimated"].astype(int) == 0
    BA = col["BA_keV"].astype(float)[keep] / 1e3
    return {
        "Z": Z[keep], "N": N[keep], "A": A[keep], "el": col["el"][keep],
        "BA": BA, "B": BA * A[keep],
        "dB": col["dBA_keV"].astype(float)[keep] * A[keep] / 1e3,
        "mass_excess": col["mass_excess_keV"].astype(float)[keep] / 1e3,
    }


def weizsacker_terms(Z, A):
    """Матрица членов формулы Вайцзеккера (без коэффициентов): строка на ядро.

    B = a_V A − a_S A^{2/3} − a_C Z(Z−1)/A^{1/3} − a_A (A−2Z)²/A + a_P δ/A^{1/2},
    δ = +1 для чётно-чётных ядер, −1 для нечётно-нечётных, 0 при нечётном A.
    """
    Z, A = np.asarray(Z, float), np.asarray(A, float)
    N = A - Z
    delta = np.where(A % 2 == 1, 0.0, np.where(Z % 2 == 0, 1.0, -1.0))
    return np.column_stack([A, -A ** (2 / 3), -Z * (Z - 1) / A ** (1 / 3),
                            -(N - Z) ** 2 / A, delta / np.sqrt(A)])


def weizsacker(Z, A, coeffs):
    """Энергия связи B(Z, A) по формуле Вайцзеккера, МэВ."""
    return weizsacker_terms(Z, A) @ np.asarray(coeffs, float)


def fit_weizsacker(Z, A, B, A_min=1):
    """Коэффициенты (a_V, a_S, a_C, a_A, a_P) методом наименьших квадратов
    по ядрам с A ≥ A_min. Формула линейна по коэффициентам, поэтому задача
    решается одним вызовом lstsq."""
    sel = np.asarray(A) >= A_min
    M = weizsacker_terms(np.asarray(Z)[sel], np.asarray(A)[sel])
    coeffs, *_ = np.linalg.lstsq(M, np.asarray(B)[sel], rcond=None)
    return coeffs


# ---------------------------------------------------------------------------
# Нерезонансные реакции заряженных частиц (тема «Скорости термоядерных реакций»)

_KEV = u.keV.to(u.erg)
_BARN = 1e-24                                     # см²
_K_B = const.k_B.cgs.value
_AMU_G = (1 * u.u).to(u.g).value
_AMU_KEV = (1 * u.u * const.c**2).to(u.keV).value


def gamow_energy(Z1, Z2, A1, A2):
    """Энергия Гамова E_G = 2μc²(παZ₁Z₂)², кэВ; A₁, A₂ — массы ядер в а. е. м."""
    mu = A1 * A2 / (A1 + A2)
    return 2 * mu * _AMU_KEV * (math.pi * const.alpha.value * Z1 * Z2) ** 2


def sigma_v(S0, Z1, Z2, A1, A2, T):
    """Скорость ⟨σv⟩ (см³/с) нерезонансной реакции при постоянном
    астрофизическом S-факторе S0 (кэВ·барн) и температуре T (К):

        ⟨σv⟩ = √(8/πμ) (kT)^{-3/2} S0 ∫ exp(−E/kT − √(E_G/E)) dE.

    Без электронного экранирования; интеграл берётся численно вокруг
    гамовского пика E₀ = (E_G (kT)²/4)^{1/3}.
    """
    kT, EG = _K_B * T / _KEV, gamow_energy(Z1, Z2, A1, A2)
    mu = A1 * A2 / (A1 + A2) * _AMU_G
    E0 = (EG * kT**2 / 4) ** (1 / 3)
    integral = quad(lambda E: math.exp(-E / kT - math.sqrt(EG / E)), 0, 60 * E0,
                    points=[E0], limit=200)[0]                       # кэВ
    return (math.sqrt(8 / (math.pi * mu)) * (kT * _KEV) ** -1.5
            * (S0 * _KEV * _BARN) * integral * _KEV)
