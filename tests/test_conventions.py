"""Проверка чисел и соглашений со страницы conventions.qmd.

Если меняются конвенции, сначала меняется этот файл: тесты фиксируют
значения страницы «Конвенции» в исполняемом виде.
"""

import math

import astropy.constants as const
import astropy.units as u
from astropy.cosmology import Planck18

from cosmevo.units import natural


def close(a, b, rel=1e-3):
    return math.isclose(a, b, rel_tol=rel)


# --- Планковские величины ---------------------------------------------------

def test_planck_masses():
    m_pl = natural((const.hbar * const.c / const.G) ** 0.5).value
    m_reduced = m_pl / math.sqrt(8 * math.pi)
    assert close(m_pl, 1.221e19)
    assert close(m_reduced, 2.435e18)


# --- Астрономические единицы (IAU 2015, юлианский год) ----------------------

def test_astronomical_units():
    assert close(const.M_sun.cgs.value, 1.988e33)
    assert close(const.L_sun.cgs.value, 3.828e33)
    assert u.yr.to(u.day) == 365.25
    assert close(u.yr.to(u.s), 3.156e7)
    assert close(u.pc.to(u.cm), 3.086e18)
    # световой год — путь света за юлианский год
    assert close(u.lyr.to(u.cm), (const.c * 1 * u.yr).cgs.value, rel=1e-9)
    assert close(u.lyr.to(u.cm), 9.461e17)
    assert close(u.lyr.to(u.pc), 0.3066)


def test_ev_in_kelvin():
    assert close((1 * u.eV / const.k_B).to(u.K).value, 1.16e4, rel=3e-3)


# --- Опорная космология -------------------------------------------------------

def test_reference_cosmology():
    assert close(Planck18.H0.value, 67.66, rel=1e-4)
    assert close(Planck18.Om0, 0.3097)
    assert Planck18.Tcmb0.value == 2.7255
    # возраст Вселенной в этой модели
    assert close(Planck18.age(0).to(u.Gyr).value, 13.79, rel=2e-3)


def test_critical_density():
    rho_c = 3 * Planck18.H0**2 / (8 * math.pi * const.G)
    assert close(rho_c.cgs.value, Planck18.critical_density0.cgs.value, rel=1e-6)
    # ρ_c = 1.878·10⁻²⁹ h² г/см³
    h = Planck18.H0.value / 100
    assert close(rho_c.cgs.value / h**2, 1.878e-29)


def test_redshift_convention():
    # 1 + z = 1/a при a(t0) = 1
    for z in (0.5, 3.0, 1100.0):
        assert close(Planck18.scale_factor(z), 1 / (1 + z), rel=1e-12)


# --- Кривизна и уравнения Эйнштейна ---------------------------------------

def test_curvature_sign_and_einstein_equations():
    # сфера: R > 0; ФЛРУ: G_00 = 3(ȧ² + κ)/a² = 8πGρ, т. е. H² = 8πGρ/3 − κ/a²
    import sympy as sp

    from cosmevo.gr import einstein, frw_metric, ricci_scalar

    th, ph, rho = sp.symbols("theta phi rho", positive=True)
    sphere = sp.diag(rho**2, rho**2 * sp.sin(th)**2)
    assert ricci_scalar(sphere, [th, ph]).is_positive

    g, x, a = frw_metric()
    G = einstein(g, x)
    k, rho_m, Gn = sp.symbols("kappa rho G", real=True)
    H2 = sp.solve(sp.Eq(G[0, 0], 8 * sp.pi * Gn * rho_m), sp.diff(a, x[0])**2)[0] / a**2
    assert sp.simplify(H2 - (8 * sp.pi * Gn * rho_m / 3 - k / a**2)) == 0


# --- Энтропия и информация ----------------------------------------------------

def test_landauer_at_room_temperature():
    e_bit = const.k_B * 300 * u.K * math.log(2)
    assert close(e_bit.to(u.J).value, 2.87e-21)
    assert close(e_bit.to(u.eV).value, 0.0179, rel=3e-3)   # три значащие цифры
