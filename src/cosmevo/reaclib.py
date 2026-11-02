"""Скорости ядерных реакций в формате JINA REACLIB и сеть реакций.

Каждая скорость в REACLIB — сумма наборов из семи коэффициентов:

    λ(T₉) = Σ_k exp(a₀ + a₁/T₉ + a₂T₉^{−1/3} + a₃T₉^{1/3} + a₄T₉ + a₅T₉^{5/3} + a₆ ln T₉),

T₉ = T/10⁹ К. Для реакции с одним участником λ — в с⁻¹, с двумя —
N_A⟨σv⟩ в см³·моль⁻¹·с⁻¹, с тремя — N_A²⟨σv⟩ в см⁶·моль⁻²·с⁻¹.
Множитель 1/n! для одинаковых участников в таблицы не входит и
учитывается здесь, в `network_rhs`. Обратные реакции (фоторасщепление
и т. п.) даны в базе отдельными наборами (флаг «v»), вычисленными из
детального равновесия.

Источник: Cyburt R. H. et al., Astrophys. J. Suppl. 189, 240 (2010).
Файл `data/reaclib_bbn.txt` — выборка для лёгких ядер (скрипт
`scripts/data/make_reaclib_bbn.py`).
"""

from collections import Counter
from dataclasses import dataclass, field
from math import factorial

import numpy as np

from cosmevo import DATA_DIR

REACLIB_BBN = DATA_DIR / "reaclib_bbn.txt"

# Глава REACLIB → число участников слева (остальные ядра в строке — продукты)
N_REACTANTS = {1: 1, 2: 1, 3: 1, 4: 2, 5: 2, 6: 2, 7: 2, 8: 3, 9: 3, 10: 4, 11: 1}


@dataclass
class Rate:
    """Одна реакция: участники, продукты и все наборы коэффициентов."""
    reactants: tuple
    products: tuple
    Q: float                                      # МэВ
    labels: set = field(default_factory=set)
    reverse: bool = False
    weak: bool = False
    coeffs: list = field(default_factory=list)

    @property
    def name(self):
        return " + ".join(self.reactants) + " → " + " + ".join(self.products)

    def __call__(self, T9):
        """Скорость при температуре T₉ (число или массив)."""
        T9 = np.asarray(T9, float)
        powers = np.array([np.ones_like(T9), 1 / T9, T9 ** (-1 / 3), T9 ** (1 / 3),
                           T9, T9 ** (5 / 3), np.log(T9)])
        return sum(np.exp(np.tensordot(a, powers, axes=1)) for a in self.coeffs)


def _float(s):
    return float(s.replace("D", "e").replace("d", "e"))


def load_reaclib(path=REACLIB_BBN):
    """Словарь {(участники, продукты): Rate}; наборы одной реакции
    (нерезонансные и резонансные) суммируются."""
    lines = open(path, encoding="ascii").read().split("\n")
    rates = {}
    i = 0
    while i + 3 < len(lines):
        if not lines[i].strip().isdigit():
            i += 1
            continue
        chapter, header = int(lines[i]), lines[i + 1]
        nuclides = [header[5 + 5 * k:10 + 5 * k].strip() for k in range(6)]
        nuclides = [x for x in nuclides if x]
        k = N_REACTANTS[chapter]
        key = (tuple(nuclides[:k]), tuple(nuclides[k:]))
        a = ([_float(lines[i + 2][13 * j:13 * (j + 1)]) for j in range(4)]       # a₀…a₃
             + [_float(lines[i + 3][13 * j:13 * (j + 1)]) for j in range(3)])    # a₄…a₆
        rate = rates.setdefault(key, Rate(*key, Q=_float(header[52:64])))
        rate.coeffs.append(np.array(a))
        rate.labels.add(header[43:47].strip())
        rate.reverse |= header[48] == "v"
        rate.weak |= header[47] == "w"
        i += 4
    return rates


class Network:
    """Сеть реакций между ядрами species: быстрые dY/dt и якобиан.

    Все наборы коэффициентов собраны в одну матрицу, поэтому скорости
    всех реакций при данной T₉ вычисляются одним матричным умножением.
    Y — числа частиц на нуклон (моль/г), rho — плотность барионов, г/см³.
    """

    def __init__(self, rates, species):
        self.rates, self.species = list(rates), tuple(species)
        index = {s: j for j, s in enumerate(self.species)}
        self.coeffs = np.vstack([a for r in self.rates for a in r.coeffs])
        self.owner = np.concatenate([[k] * len(r.coeffs) for k, r in enumerate(self.rates)])
        self.terms = []
        for r in self.rates:
            counts = Counter(r.reactants)
            norm = 1.0 / np.prod([factorial(n) for n in counts.values()])
            change = Counter(r.products)
            change.subtract(counts)
            self.terms.append(([(index[s], n) for s, n in counts.items()], len(r.reactants) - 1, norm,
                               [(index[s], c) for s, c in change.items() if c]))

    def values(self, T9):
        """Скорости всех реакций при температуре T₉."""
        powers = np.array([1.0, 1 / T9, T9 ** (-1 / 3), T9 ** (1 / 3), T9, T9 ** (5 / 3), np.log(T9)])
        return np.bincount(self.owner, weights=np.exp(self.coeffs @ powers), minlength=len(self.rates))

    def rhs(self, T9, Y, rho):
        dY = np.zeros(len(self.species))
        for lam, (reac, k, norm, change) in zip(self.values(T9), self.terms):
            flux = lam * rho**k * norm
            for j, n in reac:
                flux *= Y[j] ** n
            for j, c in change:
                dY[j] += c * flux
        return dY

    def jacobian(self, T9, Y, rho):
        """∂(dY_i/dt)/∂Y_j (нужен неявным методам: конечные разности при
        Y ~ 10⁻⁹⁰ и скоростях ~ 10¹⁴ с⁻¹ ненадёжны)."""
        J = np.zeros((len(self.species), len(self.species)))
        for lam, (reac, k, norm, change) in zip(self.values(T9), self.terms):
            base = lam * rho**k * norm
            for j_wrt, n_wrt in reac:
                dflux = base * n_wrt * Y[j_wrt] ** (n_wrt - 1)
                for j, n in reac:
                    if j != j_wrt:
                        dflux *= Y[j] ** n
                for j, c in change:
                    J[j, j_wrt] += c * dflux
        return J
