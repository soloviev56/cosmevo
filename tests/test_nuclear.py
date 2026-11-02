"""Проверки модуля cosmevo.nuclear: таблица AME2020 и формула Вайцзеккера."""

import numpy as np

import math

from cosmevo.nuclear import fit_weizsacker, gamow_energy, load_ame, sigma_v, weizsacker


def test_ame_known_values():
    d = load_ame()
    pick = lambda z, a: d["BA"][(d["Z"] == z) & (d["A"] == a)][0]
    assert abs(pick(1, 2) - 1.112283) < 1e-6
    assert abs(pick(2, 4) - 7.073916) < 1e-6
    assert abs(pick(26, 56) - 8.790356) < 1e-6
    # максимум B/A — у никеля-62
    i = np.argmax(d["BA"])
    assert (d["Z"][i], d["A"][i]) == (28, 62)


def test_weizsacker_fit():
    d = load_ame()
    c = fit_weizsacker(d["Z"], d["A"], d["B"], A_min=20)
    # коэффициенты — в привычных пределах (МэВ)
    assert 15 < c[0] < 16.5 and 16 < c[1] < 19 and 0.6 < c[2] < 0.8
    assert 21 < c[3] < 25 and 9 < c[4] < 14
    resid = d["B"] - weizsacker(d["Z"], d["A"], c)
    assert np.sqrt(np.mean(resid[d["A"] >= 20] ** 2)) < 4.0


def test_gamow_energy_pp():
    # E_G(p + p) = 2μc²(πα)² ≈ 493 кэВ
    assert abs(gamow_energy(1, 1, 1.007276, 1.007276) - 493.1) < 0.5


def test_sigma_v_gauss_and_scaling():
    """⟨σv⟩ совпадает с гауссовым приближением гамовского пика с точностью
    до поправки ~5/(12τ) и растёт с T как T^ν, ν = (τ − 2)/3."""
    k_keV = 8.617333e-8                       # кэВ/К
    Z1, Z2, A1, A2, S0, T = 1, 1, 1.007276, 1.007276, 4.01e-22, 1.57e7
    kT = k_keV * T
    E0 = (gamow_energy(Z1, Z2, A1, A2) * kT**2 / 4) ** (1 / 3)
    tau = 3 * E0 / kT
    mu = A1 * A2 / (A1 + A2) * 1.66053907e-24
    keV = 1.602176634e-9
    gauss = (math.sqrt(8 / (math.pi * mu)) * (kT * keV) ** -1.5 * S0 * keV * 1e-24
             * math.sqrt(math.pi) / 2 * (4 / math.sqrt(3)) * math.sqrt(E0 * kT) * keV
             * math.exp(-tau))
    sv = sigma_v(S0, Z1, Z2, A1, A2, T)
    assert abs(sv / gauss - 1 - 5 / (12 * tau)) < 0.01
    nu = math.log(sigma_v(S0, Z1, Z2, A1, A2, 1.01 * T) / sv) / math.log(1.01)
    assert abs(nu - (tau - 2) / 3) < 0.1


# --- ядерное статистическое равновесие ---------------------------------------

from cosmevo.nuclear import nse, nse_network


def test_nse_conservation_and_limits():
    net = nse_network()
    for T, rho, Ye in ((3.5e9, 1e8, 0.5), (9e9, 1e8, 0.5), (3.5e9, 1e8, 0.46)):
        X, _ = nse(T, rho, Ye, net)
        assert abs(X.sum() - 1) < 1e-9
        assert abs((X * net["Z"] / net["A"]).sum() - Ye) < 1e-9
    # холодно — ⁵⁶Ni при Y_e = 0,5; горячо — свободные нуклоны
    X, _ = nse(3.5e9, 1e8, 0.5, net)
    assert X[(net["Z"] == 28) & (net["A"] == 56)][0] > 0.95
    X, _ = nse(2e10, 1e8, 0.5, net)
    assert X[:2].sum() > 0.95


def test_nse_deuteron_saha():
    """Сеть n, p, d: n_d/(n_n n_p) = (3/4)(4π/(m T))^{3/2} e^{B_d/T} с m = m_u."""
    net = {k: v[:3] for k, v in nse_network().items()}
    assert list(net["A"]) == [1, 1, 2]
    T_MeV, rho = 0.1, 1e-5
    T = T_MeV * 1.160451812e10
    X, _ = nse(T, rho, 0.8, net)
    m_u_MeV, hbarc = 931.49410242, 1.97326980e-11                   # МэВ, МэВ·см
    n_B = rho / 1.66053907e-24
    n_n, n_p, n_d = X[0] * n_B, X[1] * n_B, X[2] / 2 * n_B
    expected = 0.75 * (4 * math.pi * hbarc**2 / (m_u_MeV * T_MeV)) ** 1.5 * math.exp(2.224566 / T_MeV)
    assert abs(n_d / (n_n * n_p) / expected - 1) < 1e-5            # константы округлены
