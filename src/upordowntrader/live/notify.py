"""
Tell a human by email, as PredictionArbitrageTrader does: live trading asks for one when it halts, see live/brakes.py.
Every alert is logged and stored in the alerts table as it is raised, and emailed in a background thread, so the trading
never waits on the mail server. Whether the email went out, or why not, is written back to the alert.

The mail server and addresses are in data/email.json, gitignored like the venue keys, in PredictionArbitrageTrader's
format, so its file can be copied over, and read at each alert, so it can be added without a restart:

    {"host": "smtp.gmail.com", "port": 587, "user": "me@gmail.com",
     "password": "an app password", "from": "me@gmail.com", "to": ["me@gmail.com"]}

Port 465 connects over TLS from the start, any other port upgrades with STARTTLS. Without the file the alert is still
logged and stored. `python3 -m upordowntrader.live.executor --check --email` sends a test.
"""

import asyncio
import json
import smtplib
import ssl
import time
from email.message import EmailMessage
from upordowntrader.common.log import on_failure
from upordowntrader.common.paths import DATA_DIR
from upordowntrader.db import database

EMAIL_FILE = DATA_DIR / "email.json"


def send_email(settings, subject, body):
    """
    Send one plain text email through the SMTP server the settings name.
    """
    message = EmailMessage()
    message["Subject"], message["From"], message["To"] = subject, settings["from"], ", ".join(settings["to"])
    message.set_content(body)
    port = int(settings.get("port", 587))
    context = ssl.create_default_context()
    if port == 465:
        server = smtplib.SMTP_SSL(settings["host"], port, timeout=30, context=context)
    else:
        server = smtplib.SMTP(settings["host"], port, timeout=30)
        server.starttls(context=context)
    with server:
        if settings.get("user"):
            server.login(settings["user"], settings["password"])
        server.send_message(message)


def deliver(subject, body, sender=send_email):
    """
    Email one alert through the settings in EMAIL_FILE. Returns None when it went out, or why it did not.
    """
    try:
        settings = json.loads(EMAIL_FILE.read_text())
    except FileNotFoundError:
        return f"no email settings in {EMAIL_FILE.name}"
    except (OSError, ValueError) as e:
        return f"email settings in {EMAIL_FILE.name} could not be read ({e!r})"
    try:
        sender(settings, subject, body)
    except Exception as e:
        return repr(e)
    return None


class Notifier:
    """
    Stores, logs, and emails alerts. sender is the function that sends one email, given the settings, subject, and body.
    """

    def __init__(self, conn, log=print, sender=send_email, clock=time.time):
        self.conn = conn
        self.log = log
        self.sender = sender
        self.clock = clock
        self.tasks = set()          # Emails being sent.

    def finish(self, alert_id, error):
        """
        Write back whether the alert's email went out, and log when it did not.
        """
        database.update(self.conn, "alerts", alert_id, {"sent_at": None if error else self.clock(), "error": error})
        if error:
            self.log(f"alert {alert_id} was not emailed: {error}")

    def send(self, kind, subject, body):
        """
        Raise an alert now. Inside the running loop the email goes out in a background thread, outside it at once.
        Returns the alert's id.
        """
        alert_id = database.store_alert(self.conn, {"time": self.clock(), "kind": kind, "subject": subject, "body": body,
                                                    "sent_at": None, "error": None})
        self.log(f"alert: {subject}")
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            self.finish(alert_id, deliver(subject, body, self.sender))
            return alert_id

        def sent(task):
            if not task.cancelled() and task.exception() is None:
                self.finish(alert_id, task.result())

        task = asyncio.create_task(asyncio.to_thread(deliver, subject, body, self.sender))
        self.tasks.add(task)
        task.add_done_callback(self.tasks.discard)
        task.add_done_callback(sent)
        task.add_done_callback(on_failure(self.log, "alert email"))
        return alert_id
