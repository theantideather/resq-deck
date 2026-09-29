"""Send short messages to Telegram and/or a chat webhook (Discord, Slack).

    TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID   from @BotFather and your chat
    AURUM_NOTIFY_WEBHOOK                   Discord or Slack incoming webhook URL

Nothing configured means nothing is sent. Failures are logged, never raised:
a notification problem must not stop the trading loop.
"""

from __future__ import annotations

import json
import os
import urllib.request


def _post(url: str, payload: dict) -> None:
    req = urllib.request.Request(url, data=json.dumps(payload).encode(), method="POST",
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=10) as r:
        r.read()


class Notifier:
    def __init__(self, telegram_token: str | None = None, telegram_chat: str | None = None,
                 webhook: str | None = None, sender=_post):
        self.tg_token = telegram_token if telegram_token is not None else os.environ.get("TELEGRAM_BOT_TOKEN", "")
        self.tg_chat = telegram_chat if telegram_chat is not None else os.environ.get("TELEGRAM_CHAT_ID", "")
        self.webhook = webhook if webhook is not None else os.environ.get("AURUM_NOTIFY_WEBHOOK", "")
        self._send = sender

    @property
    def enabled(self) -> bool:
        return bool((self.tg_token and self.tg_chat) or self.webhook)

    def send(self, text: str) -> bool:
        text = text[:3500]
        sent = False
        if self.tg_token and self.tg_chat:
            try:
                self._send(f"https://api.telegram.org/bot{self.tg_token}/sendMessage",
                           {"chat_id": self.tg_chat, "text": text, "disable_web_page_preview": True})
                sent = True
            except Exception as e:
                print(f"[aurum] telegram failed: {e}")
        if self.webhook:
            try:
                # Discord reads "content", Slack reads "text".
                self._send(self.webhook, {"content": text, "text": text})
                sent = True
            except Exception as e:
                print(f"[aurum] webhook notify failed: {e}")
        return sent
