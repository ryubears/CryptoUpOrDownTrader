#!/usr/bin/env bash
# What the instance is doing, in one look: run on the instance as ubuntu, or from the Mac through ssh, see commands.txt.
#     bash ~/CryptoUpOrDownTrader/deploy/check.sh                           the paper report of the last 24 hours
#     bash ~/CryptoUpOrDownTrader/deploy/check.sh --since 2026-10-15T14:15  from a time, in UTC
#     bash ~/CryptoUpOrDownTrader/deploy/check.sh --all                     from the first order
#     bash ~/CryptoUpOrDownTrader/deploy/check.sh --hours 24 --mode live    live trading alone, or --mode paper
# The code running, the services and their restarts, the disk and the database, each feed's last status line, the gaps
# of the last day, the tracebacks in the logs, the paper trader's and the live trader's last status lines and whether
# live trading has halted, then the report of paper and live trading, see paper/report.py, which takes the arguments
# given.

set -uo pipefail

DIR="$HOME/CryptoUpOrDownTrader"
LOGS="$DIR/data/logs"
DB="$DIR/data/history.sqlite"
SERVICES="crypto-up-or-down-trader crypto-up-or-down-paper crypto-up-or-down-live"
HALT="$DIR/data/live_halt.txt"

echo "== $(date -u '+%Y-%m-%d %H:%M:%S') UTC, code $(git -C "$DIR" log --oneline -1)"
for service in $SERVICES; do
    echo "$service: $(systemctl is-active "$service"), $(systemctl show "$service" -p NRestarts --value) restarts, since" \
         "$(systemctl show "$service" -p ActiveEnterTimestamp --value)"
done
df -h / | awk 'NR == 2 {print "disk: " $3 " used of " $2 ", " $4 " free"}'
echo "database: $(du -h "$DB" | cut -f1)"
free -m | awk '/Mem/ {print "memory: " $7 " MB available of " $2 " MB"}'

echo
echo "== feeds, the recorder's last status lines"
tail -n 40 "$LOGS/record-kalshi.log" | grep -a -E "tracking|quotes,|marks" | tail -n 5 | cut -c1-240
echo
echo "== gaps of the last 24 hours, by feed: how many, seconds in all"
sqlite3 -readonly "$DB" "SELECT venue, COUNT(*), ROUND(SUM(end_time - start_time), 1) FROM gaps
                         WHERE start_time > strftime('%s', 'now') - 86400 GROUP BY venue ORDER BY venue" | sed 's/^/  /'
echo
echo "== tracebacks in the logs, and the last error"
for log in record-kalshi paper live; do
    [ -f "$LOGS/$log.log" ] || continue
    # A traceback comes under its log line, indented: its frames by more than 4 spaces, its error by 4. The log's times
    # have no date.
    last=$(awk '/^[0-9][0-9]:[0-9][0-9]:[0-9][0-9] / {when = $1}
                /^ +Traceback/ {inside = 1; at = when; next}
                inside && /^    [^ ]/ {sub(/^ +/, ""); error = at " " $0; inside = 0}
                END {print error}' "$LOGS/$log.log")
    echo "  $log: $(grep -ac Traceback "$LOGS/$log.log") in all${last:+, the last at ${last:0:160}}"
done
echo
echo "== the paper trader's last status line"
grep -a -E "paper: (KXBTC|no window)" "$LOGS/paper.log" | tail -n 1 | cut -c1-300
echo
echo "== live trading's last status line, and whether it has halted"
[ -f "$LOGS/live.log" ] && grep -a -E "live: " "$LOGS/live.log" | tail -n 1 | cut -c1-300
if [ -f "$HALT" ]; then echo "HALTED, remove $HALT to resume:"; sed 's/^/  /' "$HALT"; else echo "not halted"; fi

echo
if [ "$#" -eq 0 ]; then
    set -- --hours 24
elif [ "$1" = "--all" ]; then
    shift
fi
cd "$DIR/src" && nice -n 19 python3 -m upordowntrader.paper.report "$@"
