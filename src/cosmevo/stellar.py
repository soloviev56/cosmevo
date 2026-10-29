"""Политропные модели звёзд: уравнение Лейна–Эмдена и безразмерные
коэффициенты политроп.

Политропа P = K ρ^{1+1/n} в гидростатическом равновесии: ρ = ρ_c θⁿ,
r = α ξ, α² = (n+1) K ρ_c^{1/n−1} / (4πG), и
    θ'' + (2/ξ) θ' + θⁿ = 0,   θ(0) = 1, θ'(0) = 0.
Поверхность — первый нуль ξ₁ функции θ (существует при n < 5).
Коэффициенты (Чандрасекар, 1939, гл. IV):
    ω_n = −ξ₁² θ'(ξ₁)                    — безразмерная масса,
    ρ_c / ρ̄ = ξ₁³ / (3 ω_n),
    P_c = W_n G M² / R⁴,  W_n = 1 / (4π (n+1) θ'(ξ₁)²),
    Ω = −3/(5−n) · G M² / R               — гравитационная энергия.

Вырожденный ферми-газ (T = 0, спин 1/2) любой степени релятивизма,
x = p_F/(m c) (Чандрасекар, 1939, гл. X), и равновесные конфигурации из
него: белые карлики (ньютоновская гидростатика) и звёзды из свободного
нейтронного газа (уравнение Толмена–Оппенгеймера–Волкова). Единицы СГС.
"""

import math
from dataclasses import dataclass

import astropy.constants as const
import astropy.units as u
import numpy as np
from scipy.integrate import solve_ivp

_G, _C, _HBAR = const.G.cgs.value, const.c.cgs.value, const.hbar.cgs.value
M_E, M_N = const.m_e.cgs.value, const.m_n.cgs.value
M_U = (1 * u.u).to(u.g).value


@dataclass
class Polytrope:
    n: float
    xi1: float          # первый нуль θ
    dtheta1: float      # θ'(ξ₁)
    sol: object         # решение solve_ivp с dense_output

    @property
    def omega(self):
        return -self.xi1**2 * self.dtheta1

    @property
    def central_concentration(self):
        """ρ_c / ρ̄."""
        return self.xi1**3 / (3 * self.omega)

    @property
    def W(self):
        """P_c = W G M²/R⁴."""
        return 1 / (4 * math.pi * (self.n + 1) * self.dtheta1**2)

    @property
    def potential_coefficient(self):
        """q в Ω = −q G M²/R."""
        return 3 / (5 - self.n)

    def theta(self, xi):
        """θ(ξ) при 0 ≤ ξ ≤ ξ₁ (в нуле — разложение)."""
        xi = np.asarray(xi, dtype=float)
        out = np.where(xi < self.sol.t[0], 1 - xi**2 / 6, 0.0)
        inside = xi >= self.sol.t[0]
        out[inside] = self.sol.sol(xi[inside])[0]
        return np.clip(out, 0.0, None)

    def profiles(self, num=400):
        """Безразмерные профили от центра до поверхности:
        x = r/R, ρ/ρ_c, P/P_c, m/M."""
        xi = np.linspace(0, self.xi1, num)
        th = self.theta(xi)
        dth = np.where(xi > 0, self.sol.sol(np.maximum(xi, self.sol.t[0]))[1], 0.0)
        m = -xi**2 * dth / self.omega
        return xi / self.xi1, th**self.n, th**(self.n + 1), m


def lane_emden(n, xi_max=40.0):
    """Решение уравнения Лейна–Эмдена для индекса 0 ≤ n < 5."""
    if not 0 <= n < 5:
        raise ValueError("конечный радиус только при 0 ≤ n < 5")
    rhs = lambda xi, y: [y[1], -np.abs(y[0]) ** n - 2 * y[1] / xi]
    xi0 = 1e-6                                   # разложение θ ≈ 1 − ξ²/6
    surface = lambda xi, y: y[0]
    surface.terminal, surface.direction = True, -1
    sol = solve_ivp(rhs, [xi0, xi_max], [1 - xi0**2 / 6, -xi0 / 3], events=surface,
                    dense_output=True, rtol=1e-12, atol=1e-14)
    xi1, (_, dtheta1) = sol.t_events[0][0], sol.y_events[0][0]
    return Polytrope(n, xi1, dtheta1, sol)


def _fermi_scale(m):
    """m c² / λ³, λ = ħ/(m c) — масштаб давления и плотности энергии."""
    return m * _C**2 / (_HBAR / (m * _C)) ** 3


def fermi_number_density(x, m=M_E):
    """Концентрация вырожденных фермионов, см⁻³: n = x³/(3π²λ³)."""
    return np.asarray(x, float) ** 3 * (m * _C / _HBAR) ** 3 / (3 * math.pi**2)


def fermi_pressure(x, m=M_E):
    """Давление вырожденного газа, дин/см²."""
    x = np.asarray(x, float)
    return _fermi_scale(m) / (24 * math.pi**2) * (
        x * (2 * x**2 - 3) * np.sqrt(1 + x**2) + 3 * np.arcsinh(x))


def fermi_energy_density(x, m=M_E):
    """Плотность энергии (с энергией покоя), эрг/см³."""
    x = np.asarray(x, float)
    return _fermi_scale(m) / (8 * math.pi**2) * (
        x * (2 * x**2 + 1) * np.sqrt(1 + x**2) - np.arcsinh(x))


def _dP_dx(x, m):
    return _fermi_scale(m) * x**4 / (3 * math.pi**2 * math.sqrt(1 + x**2))


def _integrate_star(rhs, x_c, density_c, r_max, max_step):
    """Интеграл от центра до x = 10⁻⁴ x_c; возвращает (M, R) в г и см."""
    r0 = 1e-6 * r_max
    surface = lambda r, y: y[0] - 1e-4 * x_c
    surface.terminal = True
    sol = solve_ivp(rhs, [r0, r_max], [x_c, 4 / 3 * math.pi * r0**3 * density_c],
                    events=surface, rtol=1e-9, atol=[1e-12 * x_c, 1e-12 * density_c * r0**3],
                    max_step=max_step)
    if not sol.t_events[0].size:
        raise RuntimeError("поверхность не достигнута")
    return sol.y[1, -1], sol.t[-1]


def white_dwarf(x_c, mu_e=2.0):
    """Масса и радиус (г, см) холодного белого карлика с центральным x_c
    (идеальный вырожденный электронный газ, ньютоновская гравитация)."""
    rho_of = lambda x: mu_e * M_U * fermi_number_density(x)

    def rhs(r, y):
        x, mass = max(y[0], 0.0), y[1]
        if x == 0:
            return [0.0, 0.0]
        return [-_G * mass * rho_of(x) / r**2 / _dP_dx(x, M_E), 4 * math.pi * r**2 * rho_of(x)]

    return _integrate_star(rhs, x_c, rho_of(x_c), r_max=1e11, max_step=1e7)


def neutron_star_ov(x_c):
    """Масса и радиус (г, см) звезды из свободного вырожденного нейтронного
    газа по уравнению Толмена–Оппенгеймера–Волкова (Оппенгеймер и Волков, 1939)."""
    def rhs(r, y):
        x, mass = max(y[0], 0.0), y[1]
        if x == 0:
            return [0.0, 0.0]
        P, eps = fermi_pressure(x, M_N), fermi_energy_density(x, M_N)
        dP_dr = (-_G * (eps + P) / _C**2 * (mass + 4 * math.pi * r**3 * P / _C**2)
                 / (r**2 * (1 - 2 * _G * mass / (r * _C**2))))
        return [dP_dr / _dP_dx(x, M_N), 4 * math.pi * r**2 * eps / _C**2]

    return _integrate_star(rhs, x_c, fermi_energy_density(x_c, M_N) / _C**2,
                           r_max=1e8, max_step=1e4)
