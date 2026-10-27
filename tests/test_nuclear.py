"""Проверки модуля cosmevo.nuclear: таблица AME2020 и формула Вайцзеккера."""

import numpy as np

from cosmevo.nuclear import fit_weizsacker, load_ame, weizsacker


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
