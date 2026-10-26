#!/usr/bin/env bash
# Установить таймер systemd пользователя для ежедневного запуска.
#   scripts/daily/install.sh            — установить и включить
#   scripts/daily/install.sh --remove   — выключить и удалить
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
UNITS="$HOME/.config/systemd/user"

if [[ "${1:-}" == "--remove" ]]; then
    systemctl --user disable --now cosmevo-daily.timer || true
    rm -f "$UNITS/cosmevo-daily.service" "$UNITS/cosmevo-daily.timer"
    systemctl --user daemon-reload
    echo "таймер удалён"
    exit 0
fi

mkdir -p "$UNITS"
cp "$HERE/cosmevo-daily.service" "$HERE/cosmevo-daily.timer" "$UNITS/"
chmod +x "$HERE/run.sh"
systemctl --user daemon-reload
systemctl --user enable --now cosmevo-daily.timer
systemctl --user list-timers cosmevo-daily.timer --no-pager
