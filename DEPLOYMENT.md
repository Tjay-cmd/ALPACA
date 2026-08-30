# Deploy the trading agent as a systemd service

The scheduler is headless: no `input()`, file paths are next to the scripts, and activity is written to `scheduler.log` plus the systemd journal.

Assumes the project lives at `/root/ALPACA` with a venv at `/root/ALPACA/venv` and a filled-in `/root/ALPACA/.env`.

## Install and enable

```bash
cd /root/ALPACA
python3 -m venv venv
./venv/bin/pip install -r requirements.txt
cp /root/ALPACA/trading-agent.service /etc/systemd/system/trading-agent.service
systemctl daemon-reload
systemctl enable trading-agent.service
systemctl start trading-agent.service
```

## Check status

```bash
systemctl status trading-agent.service
```

## Logs

File log (rotating, 10MB x 5 backups):

```bash
tail -f /root/ALPACA/scheduler.log
```

systemd journal:

```bash
journalctl -u trading-agent.service -f
```

## Stop / restart

```bash
systemctl stop trading-agent.service
systemctl restart trading-agent.service
```
