"""Тензоры кривизны метрики в SymPy.

Соглашения — как на странице «Конвенции» (и у Ландау–Лифшица):
    Γ^a_bc   = ½ g^ad (∂_b g_dc + ∂_c g_db − ∂_d g_bc),
    R^a_bcd  = ∂_c Γ^a_db − ∂_d Γ^a_cb + Γ^a_ce Γ^e_db − Γ^a_de Γ^e_cb,
    R_bd     = R^a_bad,   R = g^bd R_bd,
    G_ab     = R_ab − ½ g_ab R.
При сигнатуре (+,−,−,−) уравнения Эйнштейна: G_ab = 8πG T_ab + Λ g_ab.

Метрика — квадратная sympy.Matrix, координаты — список символов.
Функции рассчитаны на небольшие размерности (до 4) и простые метрики.
"""

import sympy as sp


def _simplify(expr):
    """simplify после раскрытия кратных углов: иначе SymPy оставляет
    выражения вроде (1 − cos 2θ − sin 2θ·tg θ), равные нулю."""
    return sp.simplify(sp.expand_trig(expr))


def christoffel(g, x):
    """Символы Кристоффеля Γ[a][b][c] = Γ^a_bc."""
    n = len(x)
    ginv = g.inv()
    return [[[_simplify(sum(ginv[a, d] * (sp.diff(g[d, c], x[b]) + sp.diff(g[d, b], x[c])
                                            - sp.diff(g[b, c], x[d])) for d in range(n)) / 2)
              for c in range(n)] for b in range(n)] for a in range(n)]


def riemann(g, x):
    """Тензор Римана R[a][b][c][d] = R^a_bcd."""
    n = len(x)
    Γ = christoffel(g, x)

    def component(a, b, c, d):
        expr = sp.diff(Γ[a][d][b], x[c]) - sp.diff(Γ[a][c][b], x[d])
        expr += sum(Γ[a][c][e] * Γ[e][d][b] - Γ[a][d][e] * Γ[e][c][b] for e in range(n))
        return _simplify(expr)

    return [[[[component(a, b, c, d) for d in range(n)] for c in range(n)]
             for b in range(n)] for a in range(n)]


def ricci(g, x):
    """Тензор Риччи R_bd = R^a_bad (матрица)."""
    n = len(x)
    R = riemann(g, x)
    return sp.Matrix(n, n, lambda b, d: sp.simplify(sum(R[a][b][a][d] for a in range(n))))


def ricci_scalar(g, x):
    """Скалярная кривизна R = g^bd R_bd."""
    ginv, Ric = g.inv(), ricci(g, x)
    n = len(x)
    return sp.simplify(sum(ginv[b, d] * Ric[b, d] for b in range(n) for d in range(n)))


def einstein(g, x):
    """Тензор Эйнштейна G_ab = R_ab − ½ g_ab R (матрица)."""
    Ric = ricci(g, x)
    ginv = g.inv()
    n = len(x)
    R = sp.simplify(sum(ginv[b, d] * Ric[b, d] for b in range(n) for d in range(n)))
    return sp.simplify(Ric - g * R / 2)


def frw_metric(kappa=None):
    """Метрика ФЛРУ (+,−,−,−) в координатах (t, r, θ, φ) и список координат.

    Возвращает (g, x, a), где a = a(t) — функция масштабного фактора.
    kappa — кривизна (символ по умолчанию).
    """
    t, r, th, ph = sp.symbols("t r theta phi", positive=True)
    if kappa is None:
        kappa = sp.Symbol("kappa", real=True)
    a = sp.Function("a")(t)
    g = sp.diag(1, -a**2 / (1 - kappa * r**2), -a**2 * r**2, -a**2 * r**2 * sp.sin(th)**2)
    return g, [t, r, th, ph], a
