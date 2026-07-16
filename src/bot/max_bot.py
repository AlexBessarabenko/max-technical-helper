"""Точка входа Max-бота: long-polling через maxapi.

Запуск: `.venv/bin/python -m src.bot.max_bot` (нужен MAX_BOT_TOKEN в .env).
Хендлеры асинхронные, но RAG/LLM-вызов синхронный — он уносится в thread
через asyncio.to_thread, чтобы не блокировать event loop.
"""

import asyncio
import logging

from maxapi import Bot, Dispatcher, F
from maxapi.filters.command import CommandStart
from maxapi.types import BotStarted, MessageCreated

from src.bot.assistant import Assistant
from src.config import get_settings
from src.observability.tracing import init_langfuse
from src.rag.chain import RAGPipeline
from src.rag.people import PeopleDirectory

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

WELCOME = ("Здравствуйте! Я внутренний ассистент «ТехноСферы». Отвечу на вопросы по IT и HR "
           "(VPN, отпуск, ДМС, доступы…) и помогу найти коллегу. Просто напишите вопрос.")

# Max ограничивает длину сообщения ~4000 символов — отвечаем с запасом.
_MAX_MESSAGE_LEN = 3900


def build_dispatcher(bot: Bot, assistant: Assistant) -> Dispatcher:
    dp = Dispatcher()

    @dp.bot_started()
    async def on_started(event: BotStarted):
        await bot.send_message(chat_id=event.chat_id, text=WELCOME)

    @dp.message_created(CommandStart())
    async def on_start(event: MessageCreated):
        await event.message.answer(WELCOME)

    @dp.message_created(F.message.body.text)
    async def on_message(event: MessageCreated):
        user_id = str(event.message.sender.user_id)
        answer = await asyncio.to_thread(assistant.reply, user_id, event.message.body.text)
        await event.message.answer(answer[:_MAX_MESSAGE_LEN])

    return dp


async def main():
    settings = get_settings()
    lf = init_langfuse(settings)
    rag = RAGPipeline(settings, lf=lf)
    assistant = Assistant(settings, rag, PeopleDirectory(settings.ad_path), lf)
    bot = Bot(token=settings.max_bot_token)
    await bot.delete_webhook()  # гарантируем polling
    log.info("Запускаю long-polling…")
    await build_dispatcher(bot, assistant).start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())
