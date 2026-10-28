#!/usr/bin/env bash
# Ежедневный запуск Claude Code: порция статей или исправления по замечаниям.
# Вызывается таймером systemd (scripts/daily/cosmevo-daily.timer) или вручную.
set -euo pipefail

REPO="$(cd "$(dirname "$0")/../.." && pwd)"
LOG_DIR="$REPO/logs"
LOCK="$LOG_DIR/.daily.lock"
mkdir -p "$LOG_DIR"

export PATH="$HOME/.npm-global/bin:$HOME/miniconda3/bin:/usr/local/bin:/usr/bin:/bin"
export LANG=C.UTF-8

# Не запускаться второй раз, если предыдущая порция ещё не закончилась.
exec 9>"$LOCK"
if ! flock -n 9; then
    echo "$(date -Is) предыдущий запуск ещё работает, пропускаю" >> "$LOG_DIR/daily.log"
    exit 0
fi

LOG="$LOG_DIR/$(date +%F_%H%M).log"
cd "$REPO"
echo "$(date -Is) старт, лог $LOG" >> "$LOG_DIR/daily.log"

# Полный поток событий пишется по ходу работы и сохраняется при обрыве.
EVENTS="${LOG%.log}.jsonl"
AFTER="${LOG%.log}.after.jsonl"

set +e
claude -p "$(cat "$REPO/scripts/daily/prompt.md")" \
    --permission-mode auto \
    --output-format stream-json --verbose \
    > "$EVENTS" 2>&1
status=$?
# Короткий контрольный запрос: загрузка лимитов подписки после запуска.
claude -p "Ответь одним словом: да" --max-turns 1 \
    --output-format stream-json --verbose > "$AFTER" 2>&1
# Итоговое сообщение — в $LOG, строка о расходе — в daily.log.
usage="$(python3 -I "$REPO/scripts/daily/usage.py" "$EVENTS" "$LOG" "$AFTER" 2>&1)"
set -e

echo "$(date -Is) завершено, код $status; $usage" >> "$LOG_DIR/daily.log"
exit $status
