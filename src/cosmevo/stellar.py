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
"""

import math
from dataclasses import dataclass

import numpy as np
from scipy.integrate import solve_ivp


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
