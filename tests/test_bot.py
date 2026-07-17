from unittest.mock import MagicMock

from src.bot.assistant import Assistant
from src.bot.session import SessionMemory


def _assistant(people_ans=None, rag_status="success"):
    a = Assistant.__new__(Assistant)
    a.people = MagicMock()
    a.people.lookup_answer.return_value = people_ans
    a.rag = MagicMock()
    res = MagicMock(status=rag_status, answer="Ответ из RAG")
    a.rag.answer_traced.return_value = res
    a.sessions = SessionMemory(max_turns=6)
    return a


def test_people_route_takes_priority():
    a = _assistant(people_ans="Иванова Анна, вн. 1001")
    assert a.reply("u1", "найди Иванову") == "Иванова Анна, вн. 1001"
    a.rag.answer_traced.assert_not_called()
    # Ответ справочника тоже попадает в историю диалога.
    assert a.sessions.history("u1") == [
        ("user", "найди Иванову"),
        ("assistant", "Иванова Анна, вн. 1001"),
    ]


def test_rag_route_and_history():
    a = _assistant(people_ans=None)
    assert a.reply("u1", "как настроить vpn?") == "Ответ из RAG"
    hist = a.sessions.history("u1")
    # Кросс-задачный контракт: история — list[tuple[role, text]].
    assert [h[0] for h in hist] == ["user", "assistant"]
    _, kwargs = a.rag.answer_traced.call_args
    assert kwargs["chat_history"] == []  # первый вопрос — истории ещё нет
    assert kwargs["user_id"] == "u1" and kwargs["session_id"] == "u1"
    a.reply("u1", "а для mac?")
    assert len(a.sessions.history("u1")) == 4
    _, kwargs = a.rag.answer_traced.call_args
    assert kwargs["chat_history"] == [
        ("user", "как настроить vpn?"),
        ("assistant", "Ответ из RAG"),
    ]


def test_rag_exception_returns_friendly_error():
    a = _assistant(people_ans=None)
    a.rag.answer_traced.side_effect = RuntimeError("boom")
    ans = a.reply("u1", "как настроить vpn?")
    assert ans.startswith("Извините, произошла техническая ошибка")


def test_session_memory_bounded():
    s = SessionMemory(max_turns=4)
    for i in range(6):
        s.append("u", "user", f"q{i}")
    assert len(s.history("u")) == 4
    # deque(maxlen) вытесняет самые старые реплики.
    assert s.history("u")[0] == ("user", "q2")


def test_clarification_follow_up_flow():
    # Диалог: уточнение → короткий ответ пользователя решается в его кругу.
    from src.rag.people import PeopleDirectory

    people = PeopleDirectory.__new__(PeopleDirectory, employees=[
        {"id": 18, "full_name": "Клавдий Степанов", "department": "Продажи",
         "division": "Продажи", "position": "Руководитель отдела «Розничные продажи»",
         "phone": "3018", "email": "k.stepanov@technosphere.example",
         "manager_id": None, "samaccountname": "k.stepanov"},
        {"id": 38, "full_name": "Клавдий Панов", "department": "Маркетинг",
         "division": "Маркетинг", "position": "PR-менеджер", "phone": "3038",
         "email": "k.panov@technosphere.example",
         "manager_id": None, "samaccountname": "k.panov"},
    ])
    a = Assistant.__new__(Assistant)
    a.people = people
    a.rag = MagicMock()
    a.sessions = SessionMemory(max_turns=6)

    first = a.reply("u1", "Кто такой клавдий")
    assert "Уточните" in first
    second = a.reply("u1", "Степанов")
    assert "Клавдий Степанов" in second and "Уточните" not in second
    a.rag.answer_traced.assert_not_called()
