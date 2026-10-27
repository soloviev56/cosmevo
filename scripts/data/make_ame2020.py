"""Извлечь из таблицы AME2020 (mass_1.mas20.txt) компактный CSV.

Источник: Wang M. et al., Chinese Phys. C 45, 030003 (2021);
файл https://www-nds.iaea.org/amdc/ame2020/mass_1.mas20.txt.
Запуск: python scripts/data/make_ame2020.py mass_1.mas20.txt data/ame2020.csv

В CSV: Z, N, A, символ элемента, энергия связи на нуклон B/A (кэВ) и её
погрешность, избыток массы (кэВ), признак estimated (значение получено
экстраполяцией, в оригинале помечено «#»).
"""

import csv
import sys


def parse(path):
    rows = []
    with open(path, encoding="ascii", errors="replace") as f:
        lines = f.readlines()
    # данные начинаются после строки с единицами (micro-u)
    start = next(i for i, s in enumerate(lines) if "(micro-u)" in s) + 1
    for s in lines[start:]:
        if len(s) < 80:
            continue
        # формат Фортрана: a1,i3,i5,i5,i5,1x,a3,a4,1x,f14.6,f12.6,f13.5,1x,f10.5,...
        n, z, a = int(s[4:9]), int(s[9:14]), int(s[14:19])
        el = s[20:23].strip()
        mexc, dmexc = s[28:42], s[42:54]
        ba, dba = s[54:67], s[68:78]
        est = "#" in mexc or "#" in ba
        clean = lambda x: float(x.replace("#", ".").strip()) if x.strip() not in ("", "*") else None
        rows.append((z, n, a, el, clean(ba), clean(dba), clean(mexc), clean(dmexc), int(est)))
    return rows


def main(src, dst):
    rows = parse(src)
    with open(dst, "w", newline="") as f:
        f.write("# AME2020: Wang M. et al., Chinese Phys. C 45, 030003 (2021)\n")
        w = csv.writer(f)
        w.writerow(["Z", "N", "A", "el", "BA_keV", "dBA_keV", "mass_excess_keV", "dmass_excess_keV", "estimated"])
        w.writerows(rows)
    print(len(rows), "нуклидов")


if __name__ == "__main__":
    main(*sys.argv[1:3])
