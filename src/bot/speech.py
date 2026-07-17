"""Распознавание голосовых сообщений через Yandex SpeechKit STT.

Синхронный вызов requests — из асинхронных хендлеров бота выносить в thread
(asyncio.to_thread). Формат oggopus: голосовые MAX — ogg с кодеком opus.
"""

import logging

import requests

log = logging.getLogger(__name__)

STT_URL = "https://stt.api.cloud.yandex.net/speech/v1/stt:recognize"


def transcribe(audio_bytes: bytes, settings) -> str | None:
    """
    Распознаёт аудио (ogg/opus) в текст. Возвращает None при любой ошибке
    (сеть, не-200 от API, пустой/неразборчивый результат) — с записью в лог.
    """
    try:
        resp = requests.post(
            STT_URL,
            params={
                "folderId": settings.yandex_folder_id,
                "lang": "ru-RU",
                "format": "oggopus",
            },
            headers={"Authorization": f"Api-Key {settings.yandex_api_key}"},
            data=audio_bytes,
            timeout=30,
        )
    except Exception:
        log.exception("SpeechKit STT недоступен")
        return None
    if resp.status_code != 200:
        log.warning("SpeechKit STT ответил %s: %s", resp.status_code, resp.text[:500])
        return None
    # Пустой result (тишина, неразборчивая речь) — тоже «не распознали».
    return resp.json().get("result") or None
