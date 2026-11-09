"""Скорости реакций гелиевого горения из базы JINA REACLIB — компактный файл.

Источник тот же, что у `make_reaclib_bbn.py`: JINA REACLIB (Cyburt R. H.
et al., Astrophys. J. Suppl. 189, 240, 2010), снимок «default2» от
2025-03-30 из пакета pynucastro. Оставляются прямые (не «v») наборы
коэффициентов α-захватов гелиевого горения: 3α → ¹²C, ¹²C(α, γ)¹⁶O,
¹⁶O(α, γ)²⁰Ne, ²⁰Ne(α, γ)²⁴Mg; формат REACLIB 2 сохраняется без изменений.
Запуск: python scripts/data/make_reaclib_he.py reaclib_default2_20250330 data/reaclib_he.txt
"""

import sys

KEEP = {("he4", "he4", "he4", "c12"), ("he4", "c12", "o16"),
        ("he4", "o16", "ne20"), ("he4", "ne20", "mg24")}


def main(src, dst):
    lines = open(src, encoding="ascii").read().split("\n")
    kept = []
    i = 0
    while i + 3 < len(lines):
        if not lines[i].strip().isdigit():
            i += 1
            continue
        header = lines[i + 1]
        nuclides = tuple(x for x in (header[5 + 5 * k:10 + 5 * k].strip() for k in range(6)) if x)
        if nuclides in KEEP and header[48] != "v":
            kept.extend(lines[i:i + 4])
        i += 4
    open(dst, "w", encoding="ascii").write("\n".join(kept) + "\n")


if __name__ == "__main__":
    main(*sys.argv[1:3])
