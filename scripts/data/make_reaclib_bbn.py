"""Скорости реакций между лёгкими ядрами из базы JINA REACLIB — компактный файл.

Источник: JINA REACLIB (Cyburt R. H. et al., Astrophys. J. Suppl. 189,
240, 2010; https://reaclib.jinaweb.org), снимок «default2» от 2025-03-30
в том виде, в каком его распространяет пакет pynucastro:
https://raw.githubusercontent.com/pynucastro/pynucastro/main/pynucastro/data/reaclib_default2_20250330
Оставляются все наборы коэффициентов, в которых участвуют только
n, p, d, t, ³He, ⁴He, ⁶Li, ⁷Li, ⁷Be (сеть первичного нуклеосинтеза);
формат REACLIB 2 сохраняется без изменений (4 строки на набор).
Запуск: python scripts/data/make_reaclib_bbn.py reaclib_default2_20250330 data/reaclib_bbn.txt
"""

import sys

LIGHT = {"n", "p", "d", "t", "he3", "he4", "li6", "li7", "be7"}


def main(src, dst):
    lines = open(src, encoding="ascii").read().split("\n")
    kept = []
    i = 0
    while i + 3 < len(lines):
        if not lines[i].strip().isdigit():
            i += 1
            continue
        header = lines[i + 1]
        nuclides = {header[5 + 5 * k:10 + 5 * k].strip() for k in range(6)} - {""}
        if nuclides <= LIGHT:
            kept.extend(lines[i:i + 4])
        i += 4
    with open(dst, "w", encoding="ascii") as f:
        f.write("\n".join(kept) + "\n")


if __name__ == "__main__":
    main(*sys.argv[1:3])
