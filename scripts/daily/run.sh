#!/usr/bin/env bash
# Ежедневный запуск Claude Code: порция статей или исправления по замечаниям.
# Вызывается таймером systemd (scripts/daily/gmph-daily.timer) или вручную.
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

set +e
claude -p "$(cat "$REPO/scripts/daily/prompt.md")" \
    --permission-mode auto \
    --output-format text \
    > "$LOG" 2>&1
status=$?
set -e

echo "$(date -Is) завершено, код $status" >> "$LOG_DIR/daily.log"
exit $status
