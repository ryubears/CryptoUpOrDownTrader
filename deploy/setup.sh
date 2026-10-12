#!/usr/bin/env bash
# Set up a fresh Ubuntu 24.04 EC2 instance to run the recorder, as ubuntu, from its home directory:
#     bash setup.sh
# Installs the system packages, clones the repo to ~/CryptoUpOrDownTrader, installs its Python packages for ubuntu, runs
# the tests, shows the clock's sync, which the recorder's arrival times rely on, and installs the service, enabled so it
# starts on boot, but not started: the Kalshi API key and the database go in data/ first, see commands.txt. Safe to run
# again, as after an update of this script.

set -euo pipefail

REPO=https://github.com/ryubears/CryptoUpOrDownTrader.git
DIR="$HOME/CryptoUpOrDownTrader"
SERVICE=crypto-up-or-down-trader

sudo apt-get update -q
sudo apt-get install -y -q python3-pip git sqlite3 chrony
[ -d "$DIR/.git" ] || git clone "$REPO" "$DIR"
cd "$DIR"
git pull --ff-only
pip3 install --user --break-system-packages -q -r requirements.txt
mkdir -p data/logs
python3 -m pytest -q
chronyc tracking | grep -E "Reference ID|System time|Leap status"
sudo cp "deploy/$SERVICE.service" /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable "$SERVICE"
echo "Set up. Put the Kalshi key files and the database in $DIR/data, then: sudo systemctl start $SERVICE"
