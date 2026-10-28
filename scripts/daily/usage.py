"""Сводка по запуску Claude Code из потока событий --output-format stream-json.

    python3 usage.py run.jsonl run.log [after.jsonl]
        → итоговое сообщение Claude — в run.log, строка о расходе — в stdout.
          after.jsonl — поток короткого контрольного запроса сразу после
          запуска: из него берётся загрузка лимитов в конце (внутри долгого
          запуска событие о лимитах может прийти только в начале).

Строка о расходе: длительность, число ходов, токены, условная стоимость по
прейскуранту API (при подписке не списывается, но сравнима между запусками)
и загрузка лимитов подписки — пятичасового окна и недели — в начале и в
конце запуска. Разница недельной загрузки — доля недели, ушедшая на запуск.
Если запуск оборвался, событие result может отсутствовать: тогда сводка
строится по тому, что успело записаться.
"""

import json
import sys
from datetime import datetime


def read_events(path):
    events = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                pass                     # строки не в JSON (сообщения об ошибках)
    return events


def windows(event):
    info = event.get("rate_limit_info", {})
    return info.get("unifiedWindows", {})


def pct(x):
    return f"{100 * x:.0f}%"


def limits_line(limit_events):
    if not limit_events:
        return "лимиты: нет данных"
    first, last = windows(limit_events[0]), windows(limit_events[-1])
    parts = []
    for key, name in (("five_hour", "5 ч"), ("seven_day", "неделя")):
        if key in first and key in last:
            reset = datetime.fromtimestamp(last[key]["resetsAt"]).strftime("%d.%m %H:%M")
            parts.append(f"{name} {pct(first[key]['utilization'])}→"
                         f"{pct(last[key]['utilization'])} (сброс {reset})")
    return "лимиты: " + ", ".join(parts)


def summary(events):
    result = next((e for e in reversed(events) if e.get("type") == "result"), None)
    limits = [e for e in events if e.get("type") == "rate_limit_event"]
    if result is None:
        turns = sum(e.get("type") == "assistant" for e in events)
        head = f"расход: запуск оборван, итога нет; ответов модели {turns}"
    else:
        u = result.get("usage", {})
        head = (f"расход: {result.get('duration_ms', 0) / 60000:.0f} мин, "
                f"ходов {result.get('num_turns')}, "
                f"токены: вход {u.get('input_tokens', 0)}, "
                f"кэш {u.get('cache_read_input_tokens', 0) + u.get('cache_creation_input_tokens', 0)}, "
                f"выход {u.get('output_tokens', 0)}; "
                f"по прейскуранту ${result.get('total_cost_usd', 0):.2f}")
        if result.get("is_error"):
            head += f"; ошибка: {result.get('subtype')}"
    return f"{head}; {limits_line(limits)}"


def final_text(events):
    result = next((e for e in reversed(events) if e.get("type") == "result"), None)
    if result and result.get("result"):
        return result["result"]
    # без события result — последний текст модели
    for e in reversed(events):
        if e.get("type") == "assistant":
            texts = [c.get("text", "") for c in e.get("message", {}).get("content", [])
                     if c.get("type") == "text"]
            if texts:
                return "\n".join(texts)
    return "(итогового сообщения нет)"


def main():
    jsonl, log = sys.argv[1], sys.argv[2]
    events = read_events(jsonl)
    with open(log, "w", encoding="utf-8") as f:
        f.write(final_text(events) + "\n")
    if len(sys.argv) > 3:            # загрузка лимитов после запуска
        events += [e for e in read_events(sys.argv[3]) if e.get("type") == "rate_limit_event"]
    print(summary(events))


if __name__ == "__main__":
    main()
