"""Проверки перевода в натуральные единицы (cosmevo.units)."""

import math

import astropy.constants as const
import astropy.units as u
import pytest

from cosmevo.units import from_natural, natural


def close(a, b, rel=1e-6):
    return math.isclose(a, b, rel_tol=rel)


def test_temperature_and_mass():
    assert close(natural(1 * u.K).to(u.eV).value, 8.617333e-5)
    assert close(natural(const.m_e).to(u.MeV).value, 0.51099895)


def test_length_and_time_are_inverse_energy():
    # ħc = 197.3269804 МэВ·фм
    assert close(natural(1 * u.fm).to(1 / u.MeV).value, 1 / 197.3269804)
    # ħ = 6.582119569e-25 ГэВ·с
    assert close(natural(1 * u.s).value, 1 / 6.582119569e-25)


def test_planck_mass():
    m_pl = (const.hbar * const.c / const.G) ** 0.5
    assert close(natural(m_pl).value, 1.22089e19, rel=1e-4)


def test_composite_units():
    # G в натуральных единицах — это 1/M_Pl²
    g = natural(const.G).value
    assert close(g, 1 / 1.22089e19**2, rel=1e-4)
    # плотность энергии: ГэВ⁴
    assert natural(1 * u.erg / u.cm**3).unit == u.GeV**4


def test_round_trip():
    for q in (3.2 * u.Mpc, 1 * u.yr, 5800 * u.K, 1 * u.Msun, 2.5 * u.g / u.cm**3):
        back = from_natural(natural(q), q.unit)
        assert close(back.value, q.value)


def test_charge_is_rejected():
    with pytest.raises(u.UnitsError):
        natural(1 * u.C)
