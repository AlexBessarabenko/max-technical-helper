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


_STEPANOV_FIXTURE = [
    {"id": 18, "full_name": "Клавдий Степанов", "department": "Продажи",
     "division": "Продажи", "position": "Руководитель отдела «Розничные продажи»",
     "phone": "3018", "email": "klavdiy.stepanov@technosphere.example",
     "manager_id": None, "samaccountname": "k.stepanov"},
    {"id": 38, "full_name": "Клавдий Панов", "department": "Маркетинг",
     "division": "Маркетинг", "position": "PR-менеджер", "phone": "3038",
     "email": "klavdiy.panov@technosphere.example",
     "manager_id": None, "samaccountname": "k.panov"},
    {"id": 69, "full_name": "Венедикт Степанов", "department": "Финансы",
     "division": "Финансы", "position": "Финансовый аналитик", "phone": "3069",
     "email": "venedikt.stepanov@technosphere.example",
     "manager_id": None, "samaccountname": "v.stepanov"},
    {"id": 91, "full_name": "Евпраксия Степанова", "department": "Продажи",
     "division": "Продажи", "position": "Менеджер по работе с клиентами",
     "phone": "3091", "email": "evpraksiya.stepanova@technosphere.example",
     "manager_id": None, "samaccountname": "e.stepanova"},
]


def test_full_name_beats_weak_fuzzy_hit():
    # Регрессия: «степанов» нечётко совпадает с «панов» (77) и раньше давало
    # Клавдию Панову лишнее «совпавшее слово» → ложное уточнение на полное ФИО.
    d = PeopleDirectory.__new__(PeopleDirectory, employees=_STEPANOV_FIXTURE)
    ans = d.lookup_answer("Клавдий степанов")
    assert ans is not None and "Клавдий Степанов" in ans
    assert "Уточните" not in ans


def test_clarification_options_and_follow_up():
    d = PeopleDirectory.__new__(PeopleDirectory, employees=_STEPANOV_FIXTURE)
    options = d.clarification_options("Кто такой клавдий")
    assert options is not None and len(options) == 2
    # Follow-up «Степанов» решается в пользу Клавдия Степанова из списка.
    ans = d.lookup_within(options, "Степанов")
    assert ans is not None and "Клавдий Степанов" in ans


def test_lookup_by_position_ceo():
    d = PeopleDirectory.__new__(PeopleDirectory, employees=[
        {"id": "c1", "full_name": "Вениамин Козлов", "department": "ТехноСфера",
         "division": "Дирекция", "position": "Генеральный директор", "phone": "2000",
         "email": "veniamin.kozlov@technosphere.example", "manager_id": None,
         "samaccountname": "v.kozlov"},
        {"id": "e2", "full_name": "Ирина Николаева", "department": "ИТ",
         "division": "Руководство", "position": "Руководитель департамента «ИТ»",
         "phone": "1671", "email": "irina.nikolaeva@technosphere.example",
         "manager_id": "c1", "samaccountname": "i.nikolaeva"},
    ])
    ans = d.lookup_answer("Кто генеральный директор ТехноСферы?")
    assert ans is not None and "Вениамин Козлов" in ans and "Генеральный директор" in ans


def test_lookup_position_ambiguous_returns_none():
    # Популярная должность у многих сотрудников — неоднозначно, отвечает RAG.
    d = PeopleDirectory.__new__(PeopleDirectory, employees=[
        {"id": f"r{i}", "full_name": f"Сотрудник{i} Иванов", "department": "HR",
         "division": "HR", "position": "Рекрутер", "phone": str(4000 + i),
         "email": f"r{i}@technosphere.example", "manager_id": None,
         "samaccountname": f"r{i}"}
        for i in range(3)
    ])
    assert d.lookup_answer("кто рекрутер?") is None


def test_real_data_ceo_question():
    # Золотой сценарий из goldens.json на реальном employees.json.
    d = PeopleDirectory("data/ad/employees.json")
    ans = d.lookup_answer("Кто генеральный директор ТехноСферы?")
    assert ans is not None and "Вениамин Козлов" in ans
