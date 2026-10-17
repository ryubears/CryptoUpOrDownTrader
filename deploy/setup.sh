#!/usr/bin/env bash
# Set up a fresh Ubuntu 24.04 EC2 instance to run the recorder and the paper trader, as ubuntu, from its home directory:
#     bash setup.sh
# Installs the system packages, clones the repo to ~/CryptoUpOrDownTrader, installs its Python packages for ubuntu, runs
# the tests, shows the clock's sync, which the recorder's arrival times rely on, and installs the services, enabled so they
# start on boot, but not started: the API keys and the database go in data/ first, see commands.txt. The live trading
# service is installed too, but not enabled: it sends real orders, so only a person starts it, see its unit file. Safe to
# run again, as after an update of this script.

set -euo pipefail

REPO=https://github.com/ryubears/CryptoUpOrDownTrader.git
DIR="$HOME/CryptoUpOrDownTrader"
SERVICES="crypto-up-or-down-trader crypto-up-or-down-paper"
LIVE=crypto-up-or-down-live

sudo apt-get update -q
sudo apt-get install -y -q python3-pip git sqlite3 chrony
[ -d "$DIR/.git" ] || git clone "$REPO" "$DIR"
cd "$DIR"
git pull --ff-only
pip3 install --user --break-system-packages -q -r requirements.txt
mkdir -p data/logs
python3 -m pytest -q
chronyc tracking | grep -E "Reference ID|System time|Leap status"
for service in $SERVICES $LIVE; do sudo cp "deploy/$service.service" /etc/systemd/system/; done
sudo systemctl daemon-reload
sudo systemctl enable $SERVICES
echo "Set up. Put the Kalshi and Polymarket US key files and the database in $DIR/data, then: sudo systemctl start $SERVICES"
