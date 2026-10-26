"""Перевод величин между СГС и натуральными единицами ħ = c = k_B = 1.

В натуральных единицах любая механическая или тепловая величина —
степень энергии: масса ~ E, длина и время ~ E⁻¹, температура ~ E.
Функция `natural` выражает размерную величину astropy в степени
выбранной единицы энергии (по умолчанию ГэВ), `from_natural` — обратно.
"""

import astropy.constants as const
import astropy.units as u

# Во что превращается одна единица СИ каждого базового измерения в
# натуральных единицах: (множитель в ГэВ^n, показатель n).
#   кг: m c² — энергия;  м: ħc/E;  с: ħ/E;  К: k_B T — энергия.
_BASES = {
    u.kg: ((1 * u.kg * const.c**2).to(u.GeV).value, 1),
    u.m: ((1 * u.m / (const.hbar * const.c)).to(1 / u.GeV).value, -1),
    u.s: ((1 * u.s / const.hbar).to(1 / u.GeV).value, -1),
    u.K: ((1 * u.K * const.k_B).to(u.GeV).value, 1),
}


def natural(q, energy=u.GeV):
    """Величину q (astropy Quantity) — в натуральных единицах, energy**n."""
    si = q.si.decompose()          # только базовые единицы СИ
    value, power = si.value, 0
    for base, exp in zip(si.unit.bases, si.unit.powers):
        if base not in _BASES:
            raise u.UnitsError(f"измерение {base} не сводится к энергии при ħ = c = k_B = 1")
        factor, n = _BASES[base]
        value *= factor**exp
        power += n * exp
    value *= si.unit.scale
    return (value * u.GeV**power).to(energy**power)


def from_natural(q, unit):
    """Обратный перевод: q в степенях энергии — в размерную единицу unit."""
    one = natural(1 * unit)                 # 1 unit в натуральных единицах
    return (q / one).decompose().value * unit


def cgs(q):
    """Числовое значение величины в единицах СГС."""
    return q.cgs.value
