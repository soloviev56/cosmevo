"""Сократить таблицу g_*(T) Сайкавы и Сираи до компактного CSV.

Источник: Saikawa K., Shirai S., JCAP 05, 035 (2018), arXiv:1803.01038;
файл https://member.ipmu.jp/satoshi.shirai/standardmodel2018.dat
(10⁴ точек, равномерно по lg T от 2·10⁻⁶ до 10¹⁷ ГэВ). Область КХД-перехода
в нём построена по решёточным расчётам уравнения состояния, отрыв
нейтрино — с учётом N_eff = 3,045.
Запуск: python scripts/data/make_gstar_lattice.py standardmodel2018.dat data/gstar_lattice.csv

Берётся каждая STEP-я точка (шаг по lg T ≈ 0,009 при STEP = 4: линейная
интерполяция по lg T отклоняется от полной таблицы не больше чем на 0,1
погрешности, указанной авторами).
"""

import csv
import sys

STEP = 4


def parse(path):
    rows = []
    with open(path, encoding="ascii") as f:
        for s in f:
            if s.startswith("#") or not s.strip():
                continue
            T, gr, dgr, gs, dgs = map(float, s.split())
            rows.append((T, gr, dgr, gs, dgs))
    return rows


def main(src, dst):
    rows = parse(src)
    kept = rows[::STEP]
    if kept[-1] != rows[-1]:
        kept.append(rows[-1])
    with open(dst, "w", newline="") as f:
        f.write("# g_*(T), g_*s(T) Стандартной модели: Saikawa K., Shirai S., "
                "JCAP 05, 035 (2018), arXiv:1803.01038\n")
        w = csv.writer(f)
        w.writerow(["T_GeV", "g_rho", "dg_rho", "g_s", "dg_s"])
        for r in kept:
            w.writerow([f"{x:.8g}" for x in r])
    print(len(rows), "→", len(kept), "точек")


if __name__ == "__main__":
    main(*sys.argv[1:3])
