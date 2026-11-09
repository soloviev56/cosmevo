"""Проверки модулей cosmevo.reaclib и cosmevo.bbn."""

import math

import numpy as np

from cosmevo.bbn import TAU_N, Q_NP, final_abundances, network, solve_bbn, weak_rates
from cosmevo.reaclib import Network, load_reaclib


def test_reaclib_known_rates():
    rates = load_reaclib()
    decay = rates[(("n",), ("p",))]
    assert abs(1 / decay(1.0) - 881.5) < 1.0                 # константа wc12: τ ≈ 881 с
    np_d = rates[(("n", "p"), ("d",))]
    assert 2e4 < np_d(1.0) < 3e4                             # N_A⟨σv⟩ ≈ 2,6·10⁴ см³/моль/с
    # обратная реакция d → n + p помечена «v», Q с обратным знаком
    rev = rates[(("d",), ("n", "p"))]
    assert rev.reverse and abs(rev.Q + np_d.Q) < 1e-9


def test_network_rhs_and_jacobian_simple():
    """Сеть из двух реакций с постоянными скоростями: d + d → n + ³He
    (λ₁, множитель 1/2! за одинаковые частицы) и n → p (λ₂)."""
    from cosmevo.reaclib import Rate
    const_rate = lambda a, b, lam: Rate(a, b, Q=0.0, coeffs=[np.array([math.log(lam), 0, 0, 0, 0, 0, 0])])
    lam1, lam2, rho = 3.0, 0.5, 2.0
    net = Network([const_rate(("d", "d"), ("n", "he3"), lam1), const_rate(("n",), ("p",), lam2)],
                  ("n", "p", "d", "he3"))
    Y = np.array([0.2, 0.5, 0.1, 0.05])
    flux = lam1 * rho * Y[2] ** 2 / 2
    expected = np.array([flux - lam2 * Y[0], lam2 * Y[0], -2 * flux, flux])
    assert np.allclose(net.rhs(1.0, Y, rho), expected)
    J = np.zeros((4, 4))
    J[2, 2], J[0, 2], J[3, 2] = -2 * lam1 * rho * Y[2], lam1 * rho * Y[2], lam1 * rho * Y[2]
    J[0, 0], J[1, 0] = -lam2, lam2
    assert np.allclose(net.jacobian(1.0, Y, rho), J)


def test_network_conserves_baryons_and_charge():
    net = network()
    Y = np.array([0.1, 0.7, 1e-3, 1e-5, 1e-5, 0.04, 1e-12, 1e-10, 1e-10])
    A = np.array([1, 1, 2, 3, 3, 4, 6, 7, 7])
    Z = np.array([0, 1, 1, 1, 2, 2, 3, 3, 4])
    dY = net.rhs(0.5, Y, 1e-5)
    assert abs(A @ dY) < 1e-9 * np.abs(dY).max()
    assert abs(Z @ dY) < 1e-9 * np.abs(dY).max()


def test_weak_rates_limits():
    assert abs(weak_rates(1e-3)[0] * TAU_N - 1) < 1e-6          # распад в вакууме
    for T in (3.0, 1.0, 0.5):
        l_np, l_pn = weak_rates(T)
        assert abs(l_pn / l_np / math.exp(-Q_NP / T) - 1) < 1e-5  # детальное равновесие


def test_bbn_standard_yields():
    """Стандартный BBN при η = 6,14·10⁻¹⁰ — в пределах точности упрощённого кода."""
    f = final_abundances(solve_bbn(6.14e-10, rtol=1e-5))
    assert 0.238 < f["Yp"] < 0.250
    assert 2.2e-5 < f["D/H"] < 2.8e-5
    assert 0.8e-5 < f["He3/H"] < 1.3e-5
    assert 3e-10 < f["Li7/H"] < 7e-10


def test_reaclib_helium_burning():
    """Выборка гелиевого горения: четыре прямые реакции; 3α из REACLIB
    совпадает с резонансной формулой через состояние Хойла с точностью ~10 %."""
    from cosmevo.reaclib import REACLIB_HE
    rates = load_reaclib(REACLIB_HE)
    assert set(rates) == {(("he4",) * 3, ("c12",)), (("he4", "c12"), ("o16",)),
                          (("he4", "o16"), ("ne20",)), (("he4", "ne20"), ("mg24",))}
    assert not any(r.reverse for r in rates.values())
    # резонансная формула: N_A²⟨ααα⟩ = 3!·3^{3/2}(2πħ²/(m_α kT))³ (Γ_rad/ħ) N_A² e^{−E_R/kT}
    hbarc, k, hbar_eVs, N_A = 197.3269804e-13, 8.617333262e-11, 6.582119569e-16, 6.02214076e23
    m_alpha, E_R, G_rad = 3727.379, 0.3794, 3.8e-3
    for T9 in (0.1, 0.2, 0.3):
        kT = k * T9 * 1e9
        lam = 6 * 3**1.5 * (2 * math.pi * hbarc**2 / (m_alpha * kT)) ** 3 * (G_rad / hbar_eVs) * N_A**2 \
            * math.exp(-E_R / kT)
        assert abs(lam / rates[(("he4",) * 3, ("c12",))](T9) - 1) < 0.1
