"""Детерминированный поиск сотрудников по справочнику AD (без векторного поиска)."""

import json
import re
from pathlib import Path
from typing import Dict, List, Optional

from rapidfuzz import fuzz, process

# Пороги нечёткого совпадения (бриф Task 7).
_SCORE_CUTOFF = 75     # минимальный score кандидата в extractOne
_SCORE_CONFIDENT = 85  # одиночное уверенное совпадение
_SCORE_MARGIN = 10     # минимальный отрыв от второго кандидата

# Слова-маркеры запроса про руководителя.
_MANAGER_WORDS = ("руководитель", "начальник", "менеджер", "шеф")

_WORD_RE = re.compile(r"[A-Za-zА-Яа-яЁё]+")
# Слова-кандидаты: с заглавной буквы ИЛИ длиной от 4 (склонённые фамилии в нижнем регистре).
_MIN_WORD_LEN = 4


class PeopleDirectory:
    """Справочник сотрудников: загружает employees.json и отвечает на запросы
    вида «найди Иванову» готовой карточкой, без LLM и векторного поиска."""

    def __new__(cls, ad_path=None, employees=None):
        # Инициализация именно в __new__: объект может создаваться напрямую
        # через __new__(cls, employees=[...]) без вызова __init__ (юнит-тесты).
        self = super().__new__(cls)
        if employees is None:
            employees = []
            if ad_path is not None:
                employees = json.loads(Path(ad_path).read_text(encoding="utf-8"))
        self._employees: List[dict] = list(employees)
        self._by_id: Dict = {e.get("id"): e for e in self._employees}
        self._names: List[str] = [e.get("full_name", "") for e in self._employees]
        # Токены ФИО (фамилия/имя/отчество) + владелец каждого токена —
        # для поиска по склонённым формам и опечаткам («Иванову» → «Иванова»).
        self._tokens: List[str] = []
        self._token_owner: List[int] = []
        for idx, name in enumerate(self._names):
            for token in name.split():
                self._tokens.append(token)
                self._token_owner.append(idx)
        return self

    def __init__(self, ad_path=None, employees=None):
        pass  # вся инициализация — в __new__

    def lookup_answer(self, query: str) -> Optional[str]:
        """Готовый ответ про сотрудника либо None, если запрос не про людей."""
        scores = self._match_scores(query)
        if not scores:
            return None
        # Сначала число совпавших слов запроса, затем лучший score.
        ranked = sorted(scores.items(), key=lambda kv: (kv[1][1], kv[1][0]), reverse=True)
        best_idx, (best, count) = ranked[0]
        second = ranked[1][1][0] if len(ranked) > 1 else 0.0
        # Точная ничья (те же score и число совпавших слов) — однофамильцы.
        tied = len(ranked) > 1 and ranked[1][1] == ranked[0][1]
        if tied or best >= _SCORE_CONFIDENT:
            confident = not tied
        elif len(ranked) == 1:
            # Одиночное слабое совпадение (75–84) одного слова — шум
            # («такой» ~ «котов»); уверенно лишь при совпадении нескольких слов.
            confident = count >= 2
        else:
            confident = best - second >= _SCORE_MARGIN
        if not confident:
            if len(ranked) == 1:
                return None
            return self._clarification(ranked)
        emp = self._employees[best_idx]
        answer = "Здравствуйте! " + self._card(emp)
        if self._asks_manager(query):
            manager = self._by_id.get(emp.get("manager_id"))
            if manager is not None:
                answer += " Руководитель: " + self._card(manager)
        answer += " Данные — из корпоративного справочника сотрудников."
        return answer

    def _match_scores(self, query: str) -> Dict[int, tuple]:
        """По каждому сотруднику: (лучший score, число совпавших слов запроса)."""
        scores: Dict[int, tuple] = {}
        for word in _candidate_words(query):
            per_word: Dict[int, float] = {}
            # 1) слово против полного ФИО (token_set_ratio): точные вхождения токена.
            hit = process.extractOne(
                word, self._names, scorer=fuzz.token_set_ratio,
                score_cutoff=_SCORE_CUTOFF, processor=str.lower,
            )
            if hit:
                _, score, idx = hit
                _keep_max(per_word, idx, score)
            # 2) слово против отдельных токенов ФИО: склонения и опечатки фамилий.
            for _, score, tok_idx in process.extract(
                word, self._tokens, scorer=fuzz.token_set_ratio,
                score_cutoff=_SCORE_CUTOFF, processor=str.lower,
            ):
                _keep_max(per_word, self._token_owner[tok_idx], score)
            for idx, score in per_word.items():
                best, count = scores.get(idx, (0.0, 0))
                scores[idx] = (max(best, score), count + 1)
        return scores

    def _clarification(self, ranked) -> str:
        best = ranked[0][1][0]
        close = [idx for idx, (s, _) in ranked if best - s < _SCORE_MARGIN][:5]
        options = [
            f"{self._employees[i]['full_name']} (отдел «{self._employees[i].get('department', '')}»)"
            for i in close
        ]
        listing = " или ".join([", ".join(options[:-1]), options[-1]])
        return ("Здравствуйте! Найдено несколько сотрудников с похожей фамилией. "
                f"Уточните, пожалуйста: {listing}?")

    @staticmethod
    def _asks_manager(query: str) -> bool:
        lowered = query.lower()
        return any(word in lowered for word in _MANAGER_WORDS)

    @staticmethod
    def _card(emp: dict) -> str:
        return (
            f"{emp.get('full_name', '')} — {emp.get('position', '')}, "
            f"отдел «{emp.get('department', '')}», департамент «{emp.get('division', '')}». "
            f"Внутренний номер: {emp.get('phone', '')}, почта: {emp.get('email', '')}."
        )


def _candidate_words(query: str) -> List[str]:
    """Слова с заглавной буквы + все слова длиной >= _MIN_WORD_LEN (без дублей)."""
    words, seen = [], set()
    for word in _WORD_RE.findall(query):
        key = word.lower()
        if key in seen or not (word[0].isupper() or len(word) >= _MIN_WORD_LEN):
            continue
        seen.add(key)
        words.append(word)
    return words


def _keep_max(scores: Dict[int, float], idx: int, score: float) -> None:
    if score > scores.get(idx, 0.0):
        scores[idx] = score
