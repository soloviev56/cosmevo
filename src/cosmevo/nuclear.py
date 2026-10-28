"""Энергии связи ядер: таблица AME2020 и формула Вайцзеккера.

Таблица `data/ame2020.csv` извлечена из оценки атомных масс AME2020
(Wang M. et al., Chinese Phys. C 45, 030003, 2021) скриптом
`scripts/data/make_ame2020.py`. Энергии — в МэВ.
"""

import numpy as np

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
