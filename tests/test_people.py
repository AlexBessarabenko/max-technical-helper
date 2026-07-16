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


def test_lookup_tie_returns_clarification():
    d = PeopleDirectory.__new__(PeopleDirectory, employees=[
        {"id": "r1", "full_name": "Алексей Родионов", "department": "ИТ",
         "division": "ИТ", "position": "Разработчик", "phone": "2001",
         "email": "a.rodionov@technosphere.example", "manager_id": None, "samaccountname": "a.rodionov"},
        {"id": "r2", "full_name": "Игорь Родионов", "department": "HR",
         "division": "HR", "position": "Рекрутер", "phone": "2002",
         "email": "i.rodionov@technosphere.example", "manager_id": None, "samaccountname": "i.rodionov"},
    ])
    ans = d.lookup_answer("найди Родионова")
    # Точная ничья по score → уточнение с обоими однофамильцами, а не чья-то карточка.
    assert ans is not None and "Уточните" in ans
    assert "Алексей Родионов" in ans and "Игорь Родионов" in ans


def test_lookup_single_weak_match_returns_none():
    d = PeopleDirectory.__new__(PeopleDirectory, employees=[
        {"id": "w1", "full_name": "Семен Тарасов", "department": "HR",
         "division": "HR", "position": "Специалист по кадрам", "phone": "2003",
         "email": "semen.tarasov@technosphere.example", "manager_id": None, "samaccountname": "s.tarasov"},
    ])
    # «смены» ~ «семен» = 80: одиночное слабое совпадение одним словом — шум, не сотрудник.
    assert d.lookup_answer("после смены пароля отваливается wi-fi") is None


def test_lookup_manager_of_ceo_has_no_manager_card():
    d = PeopleDirectory.__new__(PeopleDirectory, employees=[
        {"id": "c1", "full_name": "Вениамин Козлов", "department": "ТехноСфера",
         "division": "Дирекция", "position": "Генеральный директор", "phone": "2000",
         "email": "veniamin.kozlov@technosphere.example", "manager_id": None, "samaccountname": "v.kozlov"},
    ])
    ans = d.lookup_answer("кто руководитель Козлова?")
    assert ans is not None and "Вениамин Козлов" in ans
    assert "Руководитель:" not in ans  # manager_id=None — карточка руководителя не добавляется
