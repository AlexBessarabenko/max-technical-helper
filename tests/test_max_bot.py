"""Webhook-режим и логирование «пустых» message_created (голосовые, баг MAX #250)."""

import asyncio
import logging
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.bot.max_bot import _get_update_model_logging, run_webhook


def _settings(webhook_url="https://example.ru"):
    s = MagicMock()
    s.webhook_url = webhook_url
    s.webhook_path = "/webhook/max"
    s.webhook_secret = "secret-123"
    s.webhook_host = "0.0.0.0"
    s.webhook_port = 8080
    return s


def test_voice_stub_logged(caplog):
    # Платформа присылает message_created без message (нативное голосовое
    # для новых ботов — max-bot-api-client-ts#250): ждём None и понятный лог
    # вместо вводящего в заблуждение «неизвестный тип обновления».
    stub = {"update_type": "message_created", "timestamp": 1, "user_locale": "ru"}
    with caplog.at_level(logging.WARNING, logger="src.bot.max_bot"):
        result = asyncio.run(_get_update_model_logging(stub, MagicMock()))
    assert result is None
    assert any("без тела" in r.message for r in caplog.records)


def test_valid_update_passes_through():
    # Валидный message_created проходит в maxapi и возвращает модель события.
    event = {
        "update_type": "message_created",
        "timestamp": 1,
        "user_locale": "ru",
        "message": {
            "recipient": {"chat_id": 1, "chat_type": "dialog", "user_id": 2},
            "timestamp": 1,
            "body": {"mid": "mid.x", "seq": 1, "text": "привет"},
            "sender": {
                "user_id": 3,
                "first_name": "A",
                "last_name": "",
                "is_bot": False,
                "last_activity_time": 1,
                "name": "A",
            },
        },
    }
    bot = MagicMock()
    # enrich_event догружает чат через API — в тесте подменяем.
    bot.get_chat_by_id = AsyncMock(return_value=MagicMock())
    result = asyncio.run(_get_update_model_logging(event, bot))
    assert result is not None
    assert result.message.body.text == "привет"


def test_run_webhook_requires_url():
    s = _settings(webhook_url="")
    with pytest.raises(RuntimeError, match="WEBHOOK_URL"):
        asyncio.run(run_webhook(s, MagicMock(), MagicMock()))


def test_run_webhook_subscribes_and_serves():
    bot = MagicMock()
    bot.subscribe_webhook = AsyncMock()
    # Трейлинг-слеш в WEBHOOK_URL не должен давать двойной слеш в пути.
    s = _settings(webhook_url="https://example.ru/")
    with patch("src.bot.max_bot.AiohttpMaxWebhook") as wh_cls:
        wh = MagicMock()
        wh.run = AsyncMock()
        wh_cls.return_value = wh
        asyncio.run(run_webhook(s, bot, MagicMock()))

    bot.subscribe_webhook.assert_awaited_once()
    kwargs = bot.subscribe_webhook.call_args.kwargs
    assert kwargs["url"] == "https://example.ru/webhook/max"
    assert kwargs["secret"] == "secret-123"

    wh_cls.assert_called_once()
    assert wh_cls.call_args.kwargs["secret"] == "secret-123"
    wh.run.assert_awaited_once()
    assert wh.run.call_args.kwargs == {
        "host": "0.0.0.0",
        "port": 8080,
        "path": "/webhook/max",
    }
