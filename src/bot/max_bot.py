"""Точка входа Max-бота: long-polling через maxapi.

Запуск: `.venv/bin/python -m src.bot.max_bot` (нужен MAX_BOT_TOKEN в .env).
Хендлеры асинхронные, но RAG/LLM-вызов синхронный — он уносится в thread
через asyncio.to_thread, чтобы не блокировать event loop.
"""

import asyncio
import logging

import chromadb
from maxapi import Bot, Dispatcher, F
from maxapi.enums.parse_mode import TextFormat
from maxapi.enums.sender_action import SenderAction
from maxapi.filters.command import CommandStart
from maxapi.filters.filter import BaseFilter
from maxapi.types import BotStarted, MessageCreated
from maxapi.types.attachments.audio import Audio

from src.bot.assistant import Assistant
from src.bot.formatting import to_max_markdown
from src.bot.speech import transcribe
from src.config import Settings, get_settings
from src.observability.tracing import init_langfuse
from src.rag.chain import RAGPipeline
from src.rag.indexing import build_index
from src.rag.people import PeopleDirectory

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

WELCOME = ("Здравствуйте! Я внутренний ассистент «ТехноСферы». Отвечу на вопросы по IT и HR "
           "(VPN, отпуск, ДМС, доступы…) и помогу найти коллегу. Просто напишите вопрос — "
           "можно также отправить голосовое сообщение.")

VOICE_FAIL_TEXT = "Не удалось распознать речь, попробуйте ещё раз или напишите текстом."

# Max ограничивает длину сообщения ~4000 символов — отвечаем с запасом.
_MAX_MESSAGE_LEN = 3900

# Индикатор «печатает…» в Max гаснет через несколько секунд — обновляем.
_TYPING_INTERVAL_SEC = 4.0


class _AudioAttachmentFilter(BaseFilter):
    """Пропускает сообщения с аудио-вложением (голосовые)."""

    async def __call__(self, event: MessageCreated) -> bool:
        body = event.message.body
        return bool(body) and any(
            isinstance(a, Audio) for a in (body.attachments or [])
        )


async def _typing_loop(bot: Bot, chat_id: int) -> None:
    """
    Шлёт TYPING_ON, пока задачу не отменят (cancel после получения ответа).
    Ошибка отправки не роняет цикл — индикатор best-effort, просто выходим.
    """
    while True:
        try:
            await bot.send_action(chat_id=chat_id, action=SenderAction.TYPING_ON)
        except Exception:
            log.exception("Не удалось отправить typing-индикатор (chat_id=%s)", chat_id)
            return
        await asyncio.sleep(_TYPING_INTERVAL_SEC)


def build_dispatcher(bot: Bot, assistant: Assistant) -> Dispatcher:
    dp = Dispatcher()

    @dp.bot_started()
    async def on_started(event: BotStarted):
        await bot.send_message(chat_id=event.chat_id, text=WELCOME)

    @dp.message_created(CommandStart())
    async def on_start(event: MessageCreated):
        await event.message.answer(WELCOME)

    @dp.message_created(_AudioAttachmentFilter())
    async def on_voice(event: MessageCreated):
        """Голосовое: скачать аудио → SpeechKit STT → обычный ответ ассистента.

        Распознанный текст идёт в assistant.reply как обычная user-реплика
        (история диалога ведётся внутри Assistant).
        """
        message = event.message
        user_id = str(message.sender.user_id)
        audio = next(
            a for a in message.body.attachments if isinstance(a, Audio)
        )
        url = getattr(audio.payload, "url", None)
        if not url:
            log.warning("Голосовое без url для скачивания (user_id=%s)", user_id)
            await message.answer(VOICE_FAIL_TEXT)
            return

        typing = asyncio.create_task(_typing_loop(bot, message.recipient.chat_id))
        try:
            try:
                audio_bytes = await bot.download_bytes(url)
            except Exception:
                log.exception("Не удалось скачать голосовое (user_id=%s)", user_id)
                await message.answer(VOICE_FAIL_TEXT)
                return
            text = await asyncio.to_thread(transcribe, audio_bytes, assistant.settings)
            if text is None:
                await message.answer(VOICE_FAIL_TEXT)
                return
            answer = await asyncio.to_thread(assistant.reply, user_id, text)
        finally:
            typing.cancel()

        reply = f"Вы спросили: *{text}*\n\n{answer}"
        await message.answer(
            to_max_markdown(reply[:_MAX_MESSAGE_LEN]),
            format=TextFormat.MARKDOWN,
        )

    @dp.message_created(F.message.body.text)
    async def on_message(event: MessageCreated):
        user_id = str(event.message.sender.user_id)
        # LLM может думать 30+ секунд — показываем «печатает…», пока ждём.
        typing = asyncio.create_task(_typing_loop(bot, event.message.recipient.chat_id))
        try:
            answer = await asyncio.to_thread(assistant.reply, user_id, event.message.body.text)
        finally:
            typing.cancel()
        await event.message.answer(
            to_max_markdown(answer[:_MAX_MESSAGE_LEN]),
            format=TextFormat.MARKDOWN,
        )

    return dp


def ensure_index(settings: Settings) -> None:
    """
    Гарантирует непустой индекс ChromaDB перед стартом бота.
    Если коллекция отсутствует или пуста (например, чистый docker-volume),
    пересобирает индекс из базы знаний через build_index.
    """
    client = chromadb.PersistentClient(path=str(settings.chroma_dir))
    collection = client.get_or_create_collection(
        name=settings.chroma_collection,
        metadata={"hnsw:space": "cosine"},
    )
    count = collection.count()
    if count > 0:
        log.info("Индекс ChromaDB на месте (%d чанков), сборка не требуется.", count)
        return
    log.info("Индекс ChromaDB пуст — собираю из %s…", settings.kb_dir)
    n = build_index(settings)
    log.info("Индекс собран: %d чанков.", n)


async def main():
    settings = get_settings()
    ensure_index(settings)
    lf = init_langfuse(settings)
    rag = RAGPipeline(settings, lf=lf)
    assistant = Assistant(settings, rag, PeopleDirectory(settings.ad_path), lf)
    bot = Bot(token=settings.max_bot_token)
    await bot.delete_webhook()  # гарантируем polling
    log.info("Запускаю long-polling…")
    await build_dispatcher(bot, assistant).start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())
