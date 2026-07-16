"""Генератор синтетического справочника сотрудников (AD mock) компании «ТехноСфера».

Все данные на 100% выдуманные: имена генерируются Faker (ru_RU), email — на домене
technosphere.example. Генерация детерминирована по seed.

Запуск: `python -m src.data_gen.ad_gen` — пишет data/ad/employees.json.
"""

import json
import random
from pathlib import Path

from faker import Faker

COMPANY = "ТехноСфера"
EMAIL_DOMAIN = "technosphere.example"
OUTPUT_PATH = Path("data/ad/employees.json")

DEPARTMENTS = ["ИТ", "HR", "Финансы", "Продажи", "Маркетинг", "Производство"]

DIVISIONS = {
    "ИТ": ["Техподдержка", "Инфраструктура", "Разработка"],
    "HR": ["Кадры", "Рекрутинг", "Компенсации и льготы"],
    "Финансы": ["Бухгалтерия", "Финансовый контроль", "Казначейство"],
    "Продажи": ["Оптовые продажи", "Розничные продажи", "Ключевые клиенты"],
    "Маркетинг": ["Реклама", "Аналитика", "PR"],
    "Производство": ["Сборка", "Контроль качества", "Логистика"],
}

POSITIONS = {
    "ИТ": ["Инженер техподдержки", "Системный администратор", "Разработчик", "Тестировщик", "DevOps-инженер"],
    "HR": ["HR-менеджер", "Рекрутер", "Специалист по кадрам", "Специалист по компенсациям"],
    "Финансы": ["Бухгалтер", "Финансовый аналитик", "Экономист"],
    "Продажи": ["Менеджер по продажам", "Менеджер по работе с клиентами", "Торговый представитель"],
    "Маркетинг": ["Маркетолог", "Аналитик", "SMM-менеджер", "PR-менеджер"],
    "Производство": ["Инженер производства", "Технолог", "Оператор линии", "Специалист по логистике", "Контролёр качества"],
}

_TRANSLIT = {
    "а": "a", "б": "b", "в": "v", "г": "g", "д": "d", "е": "e", "ё": "e",
    "ж": "zh", "з": "z", "и": "i", "й": "y", "к": "k", "л": "l", "м": "m",
    "н": "n", "о": "o", "п": "p", "р": "r", "с": "s", "т": "t", "у": "u",
    "ф": "f", "х": "h", "ц": "c", "ч": "ch", "ш": "sh", "щ": "sch",
    "ъ": "", "ы": "y", "ь": "", "э": "e", "ю": "yu", "я": "ya",
}


def _translit(text: str) -> str:
    return "".join(_TRANSLIT.get(ch, ch) for ch in text.lower())


def _unique(base: str, taken: set) -> str:
    candidate, i = base, 2
    while candidate in taken:
        candidate = f"{base}{i}"
        i += 1
    taken.add(candidate)
    return candidate


def _role_specs(n: int) -> list:
    """Список (department, division, position, manager_id) слоями: CEO → департаменты → отделы → сотрудники.

    manager_id — 1-based id руководителя, всегда меньше id самого сотрудника,
    поэтому граф менеджеров ацикличен, а без менеджера остаётся ровно один CEO.
    """
    specs = [(COMPANY, "Дирекция", "Генеральный директор", None)]
    dept_head_id = {}
    for dept in DEPARTMENTS:
        dept_head_id[dept] = len(specs) + 1
        specs.append((dept, "Руководство", f"Руководитель департамента «{dept}»", 1))
    div_head_id = {}
    for dept in DEPARTMENTS:
        for div in DIVISIONS[dept]:
            div_head_id[(dept, div)] = len(specs) + 1
            specs.append((dept, div, f"Руководитель отдела «{div}»", dept_head_id[dept]))
    all_divs = [(dept, div) for dept in DEPARTMENTS for div in DIVISIONS[dept]]
    for i in range(max(0, n - len(specs))):
        dept, div = all_divs[i % len(all_divs)]
        specs.append((dept, div, random.choice(POSITIONS[dept]), div_head_id[(dept, div)]))
    return specs[:n]


def generate_employees(n: int = 120, seed: int = 42) -> list:
    """Генерирует n синтетических сотрудников; результат детерминирован по seed."""
    fake = Faker("ru_RU")
    Faker.seed(seed)
    random.seed(seed)

    specs = _role_specs(n)
    phones = random.sample(range(1000, 2000), n)  # внутренний номер 1xxx
    emails_taken: set = set()
    sam_taken: set = set()
    employees = []
    for idx, (department, division, position, manager_id) in enumerate(specs):
        if random.random() < 0.5:
            name, surname = fake.first_name_male(), fake.last_name_male()
        else:
            name, surname = fake.first_name_female(), fake.last_name_female()
        name_t, surname_t = _translit(name), _translit(surname)
        email_slug = _unique(f"{name_t}.{surname_t}", emails_taken)
        sam = _unique(f"{name_t[0]}.{surname_t}", sam_taken)
        employees.append({
            "id": idx + 1,
            "name": name,
            "surname": surname,
            "full_name": f"{name} {surname}",
            "department": department,
            "division": division,
            "position": position,
            "phone": str(phones[idx]),
            "email": f"{email_slug}@{EMAIL_DOMAIN}",
            "manager_id": manager_id,
            "samaccountname": sam,
        })
    return employees


if __name__ == "__main__":
    employees = generate_employees()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(employees, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Сгенерировано {len(employees)} сотрудников -> {OUTPUT_PATH}")
