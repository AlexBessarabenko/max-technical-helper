"""Диалоговая логика бота: маршрутизация в справочник сотрудников или RAG."""

import logging

from src.bot.session import SessionMemory

log = logging.getLogger(__name__)

ERROR_TEXT = (
    "Извините, произошла техническая ошибка. Попробуйте ещё раз или обратитесь "
    "в поддержку: helpdesk@technosphere.example, вн. 1001."
)


class Assistant:
    """
    reply(user_id, text):
    1) people.lookup_answer(text) — детерминированный ответ справочника,
       если запрос про сотрудника;
    2) иначе rag.answer_traced(...) с историей диалога.
    Оба маршрута пишут пару (user, assistant) в SessionMemory.
    Ошибка RAG не роняет бота — пользователь получает ERROR_TEXT.
    """

    def __init__(self, settings, rag, people, lf=None, max_turns: int = 6):
        self.settings = settings
        self.rag = rag
        self.people = people
        self.lf = lf
        self.sessions = SessionMemory(max_turns=max_turns)

    def reply(self, user_id: str, text: str) -> str:
        user_id = str(user_id)
        answer = self.people.lookup_answer(text)
        if answer is None:
            try:
                result = self.rag.answer_traced(
                    text,
                    chat_history=self.sessions.history(user_id),
                    user_id=user_id,
                    session_id=user_id,
                )
                answer = result.answer
            except Exception:
                log.exception("RAG не ответил (user_id=%s)", user_id)
                answer = ERROR_TEXT
        self.sessions.append(user_id, "user", text)
        self.sessions.append(user_id, "assistant", answer)
        return answer
