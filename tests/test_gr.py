"""Тесты тензоров кривизны: известные метрики и знаки конвенций."""

import sympy as sp

from cosmevo.gr import christoffel, einstein, frw_metric, riemann, ricci_scalar


def test_flat_space_has_no_curvature():
    x, y = sp.symbols("x y")
    g = sp.diag(1, 1)
    assert all(G == 0 for row in christoffel(g, [x, y]) for col in row for G in col)
    assert ricci_scalar(g, [x, y]) == 0


def test_two_sphere():
    # сфера радиуса ρ (положительно определённая метрика): R = 2/ρ²
    th, ph, rho = sp.symbols("theta phi rho", positive=True)
    g = sp.diag(rho**2, rho**2 * sp.sin(th)**2)
    assert sp.simplify(ricci_scalar(g, [th, ph]) - 2 / rho**2) == 0
    # R^θ_φθφ = sin²θ
    assert sp.simplify(riemann(g, [th, ph])[0][1][0][1] - sp.sin(th)**2) == 0


def test_maximally_symmetric_three_space():
    # γ_ij = dr²/(1 − κr²) + r²dΩ²: скалярная кривизна 6κ
    r, th, ph, k = sp.symbols("r theta phi kappa", positive=True)
    gamma = sp.diag(1 / (1 - k * r**2), r**2, r**2 * sp.sin(th)**2)
    assert sp.simplify(ricci_scalar(gamma, [r, th, ph]) - 6 * k) == 0


def test_frw_einstein_tensor_sign():
    # G_00 = 3(ȧ² + κ)/a²: при G_ab = 8πG T_ab и T_00 = ρ получаем
    # первое уравнение Фридмана H² = 8πGρ/3 − κ/a² (знак ρ > 0 — притяжение)
    g, x, a = frw_metric()
    t = x[0]
    k = sp.Symbol("kappa", real=True)
    G = einstein(g, x)
    adot = sp.diff(a, t)
    assert sp.simplify(G[0, 0] - 3 * (adot**2 + k) / a**2) == 0
    # G^r_r = (2aä + ȧ² + κ)/a² — пространственная компонента со смешанными индексами
    Gmixed = sp.simplify(g.inv()[1, 1] * G[1, 1])
    assert sp.simplify(Gmixed - (2 * a * sp.diff(a, t, 2) + adot**2 + k) / a**2) == 0


def test_de_sitter_solves_einstein_with_lambda():
    # a = e^{Ht}, κ = 0: G_ab = Λ g_ab при Λ = 3H²
    t, r, th, ph, H = sp.symbols("t r theta phi H", positive=True)
    a = sp.exp(H * t)
    g = sp.diag(1, -a**2, -a**2 * r**2, -a**2 * r**2 * sp.sin(th)**2)
    G = einstein(g, [t, r, th, ph])
    assert sp.simplify(G - 3 * H**2 * g) == sp.zeros(4, 4)


def test_constant_curvature_riemann_all_components():
    # R_ijkl = κ(γ_ik γ_jl − γ_il γ_jk) для всех компонент (включая угловые,
    # где без раскрытия кратных углов simplify не находит нуля)
    r, th, ph = sp.symbols("r theta phi", positive=True)
    k = sp.Symbol("kappa", real=True)
    gamma = sp.diag(1 / (1 - k * r**2), r**2, r**2 * sp.sin(th)**2)
    Rup = riemann(gamma, [r, th, ph])
    for a in range(3):
        for b in range(3):
            for c in range(3):
                for d in range(3):
                    low = sum(gamma[a, e] * Rup[e][b][c][d] for e in range(3))
                    expected = k * (gamma[a, c] * gamma[b, d] - gamma[a, d] * gamma[b, c])
                    assert sp.simplify(low - expected) == 0
