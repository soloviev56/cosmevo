"""Мировое потребление первичной энергии по годам — компактный CSV.

Источник: Our World in Data, «Global primary energy consumption by
source» (https://ourworldindata.org/grapher/global-primary-energy), по
данным Smil V. (2017) «Energy Transitions: Global and National
Perspectives» (до 1965 г., включая традиционную биомассу) и Energy
Institute Statistical Review of World Energy (с 1965 г.). Электроэнергия
неископаемых источников пересчитана методом замещения (как если бы её
получали на тепловых станциях), поэтому суммы на несколько процентов
больше итогов, которые Energy Institute даёт в своих сводках.
Запуск: python scripts/data/make_world_energy.py global-primary-energy.csv data/world_primary_energy.csv
(исходный файл: тот же адрес с суффиксом .csv?csvType=full&useColumnShortNames=true)
"""

import csv
import sys


def main(src, dst):
    with open(src, encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if r["entity"] == "World"]
    with open(dst, "w", encoding="utf-8", newline="") as f:
        f.write("# Мировое потребление первичной энергии, ТВт·ч/год: Our World in Data "
                "(Smil 2017; Energy Institute Statistical Review), выгрузка 2026-11-01\n")
        w = csv.writer(f)
        w.writerow(["year", "TWh"])
        for r in rows:
            total = sum(float(v) for k, v in r.items() if k.endswith("_twh") and v)
            w.writerow([r["year"], f"{total:.0f}"])


if __name__ == "__main__":
    main(*sys.argv[1:3])
