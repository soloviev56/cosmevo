"""Собрать карту тем пособия из topics.yml и статей.

topics.yml — единый источник правды о темах: название, уровень, статус,
зависимости, содержание. Статья темы `foo` раздела `methods/cosmology` лежит в
methods/cosmology/foo.qmd (имя файла произвольное) и указывает в заголовке
только `topic: foo`; уровень, статус и зависимости берутся из topics.yml.

Скрипт проверяет карту (неизвестные темы, циклы, зависимость от темы
более высокого уровня, неизвестный статус, статьи вне своего раздела) и пишет:
  _generated/topics.json — данные для фильтра filters/topics.lua;
  _generated/map.md      — карта тем для страницы map.qmd.
Запускается Quarto перед каждой сборкой (project.pre-render).
"""

import json
import pathlib
import sys

import yaml

ROOT = pathlib.Path(__file__).resolve().parent.parent
PLAN = ROOT / "topics.yml"
OUT = ROOT / "_generated"
AXES = ("methods", "evolution")
LEVELS = range(1, 4)
LEVEL_NAMES = {1: "основы", 2: "углублённый", 3: "исследовательский"}
# Статус центральных утверждений темы, см. conventions.qmd
STATUSES = {"У": "установлено", "Г": "гипотеза", "С": "спекуляция"}
TOPIC_FIELDS = ("id", "title", "level", "status", "requires", "about")
FORBIDDEN_IN_ARTICLE = ("level", "status", "requires")


def load_plan(path=PLAN):
    return yaml.safe_load(path.read_text(encoding="utf-8"))["sections"]


def front_matter(path):
    text = path.read_text(encoding="utf-8")
    if not text.startswith("---"):
        return {}
    return yaml.safe_load(text.split("---", 2)[1]) or {}


def scan_articles(root=ROOT):
    """Пары (путь относительно root, заголовок) для всех статей."""
    articles = []
    for axis in AXES:
        for path in sorted((root / axis).rglob("*.qmd")):
            if path.name.startswith("_") or path.name == "index.qmd":
                continue
            articles.append((path.relative_to(root).as_posix(), front_matter(path)))
    return articles


def build(sections, articles):
    """Собрать темы и список ошибок. Чистая функция — удобно для тестов."""
    topics, errors = {}, []
    for section in sections:
        sid = section["id"]
        if sid.split("/")[0] not in AXES:
            errors.append(f"раздел '{sid}': путь должен начинаться с {' или '.join(AXES)}/")
        for t in section["topics"]:
            missing = [k for k in TOPIC_FIELDS if k not in t]
            if missing:
                errors.append(f"тема {t.get('id', '?')}: нет полей {', '.join(missing)}")
                continue
            if t["id"] in topics:
                errors.append(f"тема '{t['id']}' объявлена дважды")
                continue
            if t["level"] not in LEVELS:
                errors.append(f"тема '{t['id']}': уровень {t['level']} вне диапазона 1–3")
            if t["status"] not in STATUSES:
                errors.append(f"тема '{t['id']}': статус '{t['status']}' не из {', '.join(STATUSES)}")
            topics[t["id"]] = {
                "title": t["title"],
                "section": sid,
                "level": t["level"],
                "level_name": LEVEL_NAMES.get(t["level"], "?"),
                "status": t["status"],
                "status_name": STATUSES.get(t["status"], "?"),
                "about": t["about"],
                "requires": list(t["requires"] or []),
                "required_by": [],
            }

    for tid, t in topics.items():
        for req in t["requires"]:
            if req not in topics:
                errors.append(f"тема '{tid}': требуется неизвестная тема '{req}'")
                continue
            topics[req]["required_by"].append(tid)
            if topics[req]["level"] > t["level"]:
                errors.append(
                    f"тема '{tid}' (уровень {t['level']}) требует '{req}' "
                    f"(уровень {topics[req]['level']})"
                )
    errors += find_cycles(topics)

    for rel, meta in articles:
        tid = meta.get("topic")
        if tid is None:
            errors.append(f"{rel}: в заголовке нет поля topic")
            continue
        if tid not in topics:
            errors.append(f"{rel}: темы '{tid}' нет в topics.yml")
            continue
        t = topics[tid]
        if not rel.startswith(t["section"] + "/"):
            errors.append(f"{rel}: тема '{tid}' относится к разделу {t['section']}")
        if "href" in t:
            errors.append(f"{rel}: у темы '{tid}' уже есть статья {t['href']}")
        extra = [k for k in FORBIDDEN_IN_ARTICLE if k in meta]
        if extra:
            errors.append(f"{rel}: поля {', '.join(extra)} задаются в topics.yml, а не в статье")
        t["href"] = "/" + rel
    return topics, errors


def find_cycles(topics):
    state, errors = {}, []

    def visit(tid, path):
        state[tid] = "open"
        for req in topics[tid]["requires"]:
            if req not in topics:
                continue
            if state.get(req) == "open":
                cycle = path[path.index(req):] + [req]
                errors.append("цикл зависимостей: " + " → ".join(cycle))
            elif req not in state:
                visit(req, path + [req])
        state[tid] = "done"

    for tid in topics:
        if tid not in state:
            visit(tid, [tid])
    return errors


# --- Страница карты --------------------------------------------------------

MERMAID_CLASSES = [
    "  classDef done fill:#d1e7dd,stroke:#198754,color:#0f5132",
    "  classDef todo fill:#f8f9fa,stroke:#adb5bd,color:#495057",
    "  classDef ext fill:#ffffff,stroke:#adb5bd,stroke-dasharray:4 3,color:#6c757d",
]


def node_id(tid):
    return tid.replace("-", "_")


def page_link(href):
    """Ссылка на HTML-страницу относительно корня сайта (для узлов mermaid)."""
    return href.lstrip("/").removesuffix(".qmd") + ".html"


def section_diagram(section, topics):
    own = [t["id"] for t in section["topics"]]
    external = sorted({r for tid in own for r in topics[tid]["requires"] if r not in own})
    lines = ["```{mermaid}", "flowchart TD", *MERMAID_CLASSES]
    for tid in own:
        t = topics[tid]
        cls = "done" if "href" in t else "todo"
        lines.append(f'  {node_id(tid)}["{t["title"]}<br/><small>ур. {t["level"]} · {t["status"]}</small>"]:::{cls}')
        if "href" in t:
            lines.append(f'  click {node_id(tid)} "{page_link(t["href"])}"')
    for tid in external:
        lines.append(f'  {node_id(tid)}["{topics[tid]["title"]}"]:::ext')
    for tid in own:
        for req in topics[tid]["requires"]:
            lines.append(f"  {node_id(req)} --> {node_id(tid)}")
    lines.append("```")
    return lines


def md_escape(text):
    """Экранировать символы, которые Markdown внутри таблицы примет за разметку."""
    return text.replace("|", "\\|").replace("*", "\\*")


def title_link(t):
    return f"[{t['title']}]({t['href']})" if "href" in t else t["title"]


def section_table(section, topics):
    lines = ["| Тема | Ур. | Статус | Содержание |", "|---|:-:|:-:|---|"]
    for item in section["topics"]:
        t = topics[item["id"]]
        mark = "✓ " if "href" in t else ""
        lines.append(f"| {mark}{title_link(t)} | {t['level']} | {t['status']} | {md_escape(t['about'])} |")
    return lines


def map_markdown(sections, topics):
    done = sum("href" in t for t in topics.values())
    lines = [
        f"Всего тем: **{len(topics)}**, написано: **{done}** (✓, зелёные узлы). "
        "Пунктиром показаны темы других разделов, на которые опирается раздел.",
        "",
        "## По уровням",
        "",
    ]
    for level in LEVELS:
        items = [t for t in topics.values() if t["level"] == level]
        if items:
            lines.append(f"**Уровень {level} — {LEVEL_NAMES[level]}** ({len(items)}): " + "; ".join(title_link(t) for t in items) + ".")
            lines.append("")
    axis_titles = {"methods": "Аппарат", "evolution": "Космическая эволюция"}
    current_axis = None
    for section in sections:
        axis = section["id"].split("/")[0]
        if axis != current_axis:
            lines += [f"## {axis_titles[axis]}", ""]
            current_axis = axis
        lines += [f"### {section['title']}", ""]
        lines += section_diagram(section, topics) + [""]
        lines += section_table(section, topics) + [""]
    return "\n".join(lines)


def main():
    sections = load_plan()
    topics, errors = build(sections, scan_articles())
    if errors:
        print("Ошибки в карте тем:", *errors, sep="\n  ", file=sys.stderr)
        sys.exit(1)
    OUT.mkdir(exist_ok=True)
    (OUT / "topics.json").write_text(json.dumps(topics, ensure_ascii=False, indent=1), encoding="utf-8")
    (OUT / "map.md").write_text(map_markdown(sections, topics), encoding="utf-8")
    done = sum("href" in t for t in topics.values())
    print(f"Карта тем: {len(topics)} тем, написано {done}")


if __name__ == "__main__":
    main()
