from src.rag.people import PeopleDirectory


def _dir():
    return PeopleDirectory.__new__(PeopleDirectory, employees=[
        {"id": "e1", "full_name": "Иванова Анна Сергеевна", "department": "Техподдержка",
         "division": "ИТ", "position": "Специалист поддержки", "phone": "1001",
         "email": "anna.ivanova@technosphere.example", "manager_id": "e2", "samaccountname": "a.ivanova"},
        {"id": "e2", "full_name": "Петров Пётр Иванович", "department": "ИТ",
         "division": "ИТ", "position": "Директор по ИТ", "phone": "1000",
         "email": "petr.petrov@technosphere.example", "manager_id": None, "samaccountname": "p.petrov"},
    ])


def test_lookup_by_surname_with_typo():
    d = _dir()
    ans = d.lookup_answer("как найти Иванову?")
    assert ans is not None and "Иванова Анна Сергеевна" in ans and "1001" in ans


def test_lookup_manager():
    d = _dir()
    ans = d.lookup_answer("кто руководитель Ивановой?")
    assert ans is not None and "Петров Пётр Иванович" in ans


def test_lookup_no_match_returns_none():
    assert _dir().lookup_answer("сколько дней отпуска?") is None
