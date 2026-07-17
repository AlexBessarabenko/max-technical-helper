"""Детерминированный поиск сотрудников по справочнику AD (без векторного поиска)."""

import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from rapidfuzz import fuzz, process

# Пороги нечёткого совпадения (бриф Task 7).
_SCORE_CUTOFF = 75     # минимальный score кандидата в extractOne
_SCORE_CONFIDENT = 85  # одиночное уверенное совпадение
_SCORE_MARGIN = 10     # минимальный отрыв от второго кандидата

# Слова-маркеры запроса про руководителя.
_MANAGER_WORDS = ("руководитель", "начальник", "менеджер", "шеф")

# Префикс ответа-уточнения — Assistant узнаёт его для разрешения follow-up.
CLARIFICATION_PREFIX = "Здравствуйте! Найдено несколько сотрудников с похожей фамилией."

_WORD_RE = re.compile(r"[A-Za-zА-Яа-яЁё]+")
# Слова-кандидаты: с заглавной буквы ИЛИ длиной от 4 (склонённые фамилии в нижнем регистре).
_MIN_WORD_LEN = 4


class PeopleDirectory:
    """Справочник сотрудников: загружает employees.json и отвечает на запросы
    вида «найди Иванову» или «кто генеральный директор?» готовой карточкой,
    без LLM и векторного поиска."""

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
        self._positions: List[str] = [e.get("position", "") for e in self._employees]
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
        """Готовый ответ про сотрудника либо None, если запрос не про людей.

        Сначала поиск по ФИО, затем — по должности («кто генеральный
        директор?»): персональные запросы всегда в приоритете.
        """
        return self._evaluate(query)[0] or self._lookup_by_position(query)

    def clarification_options(self, query: str) -> Optional[List[int]]:
        """Индексы сотрудников, предложенных в ответе-уточнении на query.

        None, если запрос не приводит к уточнению. Используется Assistant'ом,
        чтобы понять follow-up («Степанов» после списка кандидатов).
        """
        return self._evaluate(query)[1]

    def lookup_within(self, allowed: List[int], query: str) -> Optional[str]:
        """Поиск внутри заданного круга кандидатов (ответ на уточнение)."""
        return self._evaluate(query, allowed=set(allowed))[0]

    def _evaluate(
        self, query: str, allowed: Optional[Set[int]] = None
    ) -> Tuple[Optional[str], Optional[List[int]]]:
        """Общая логика оценки: (ответ | None, индексы кандидатов уточнения | None)."""
        scores = self._match_scores(query, allowed)
        if not scores:
            return None, None
        # Ранжируем по сумме score всех слов: два точных совпадения (100+100)
        # сильнее одного точного и одного случайного нечёткого (100+77).
        ranked = sorted(
            scores.items(), key=lambda kv: (kv[1][2], kv[1][0], kv[1][1]), reverse=True
        )
        best_idx, (best, count, total) = ranked[0]
        second_total = ranked[1][1][2] if len(ranked) > 1 else 0.0
        # Точная ничья (те же сумма, score и число совпавших слов) — однофамильцы.
        tied = len(ranked) > 1 and ranked[1][1] == ranked[0][1]
        if tied or best >= _SCORE_CONFIDENT:
            confident = not tied
        elif len(ranked) == 1:
            # Одиночное слабое совпадение (75–84) одного слова — шум
            # («такой» ~ «котов»); уверенно лишь при совпадении нескольких слов.
            confident = count >= 2
        else:
            confident = total - second_total >= _SCORE_MARGIN
        if not confident:
            if len(ranked) == 1:
                return None, None
            ids = self._clarify_ids(ranked)
            return self._clarification(ids), ids
        emp = self._employees[best_idx]
        answer = "Здравствуйте!\n" + self._card(emp)
        if self._asks_manager(query):
            manager = self._by_id.get(emp.get("manager_id"))
            if manager is not None:
                answer += "\n\nРуководитель:\n" + self._card(manager)
        answer += "\n\nДанные — из корпоративного справочника сотрудников."
        return answer, None

    def _lookup_by_position(self, query: str) -> Optional[str]:
        """Ответ на вопрос про должность («кто генеральный директор?»).

        Срабатывает только на уверенное единственное совпадение: популярные
        должности (рекрутер, бухгалтер) неоднозначны — для них отвечает RAG.
        """
        hits = process.extract(
            query, self._positions, scorer=fuzz.token_set_ratio,
            score_cutoff=_SCORE_CUTOFF, processor=str.lower,
        )
        if not hits:
            return None
        hits.sort(key=lambda h: h[1], reverse=True)
        _, best, best_idx = hits[0]
        second = hits[1][1] if len(hits) > 1 else 0.0
        if best < _SCORE_CONFIDENT or best - second < _SCORE_MARGIN:
            return None
        return ("Здравствуйте!\n" + self._card(self._employees[best_idx])
                + "\n\nДанные — из корпоративного справочника сотрудников.")

    def _match_scores(self, query: str, allowed: Optional[Set[int]] = None) -> Dict[int, tuple]:
        """По каждому сотруднику: (лучший score, число совпавших слов, сумма score).

        allowed — ограничение круга кандидатов (продолжение уточнения).
        """
        scores: Dict[int, tuple] = {}
        for word in _candidate_words(query):
            per_word: Dict[int, float] = {}
            # 1) слово против полного ФИО (token_set_ratio): точные вхождения токена.
            hit = process.extractOne(
                word, self._names, scorer=fuzz.token_set_ratio,
                score_cutoff=_SCORE_CUTOFF, processor=str.lower,
            )
            if hit and (allowed is None or hit[2] in allowed):
                _, score, idx = hit
                _keep_max(per_word, idx, score)
            # 2) слово против отдельных токенов ФИО: склонения и опечатки фамилий.
            for _, score, tok_idx in process.extract(
                word, self._tokens, scorer=fuzz.token_set_ratio,
                score_cutoff=_SCORE_CUTOFF, processor=str.lower,
            ):
                owner = self._token_owner[tok_idx]
                if allowed is not None and owner not in allowed:
                    continue
                _keep_max(per_word, owner, score)
            for idx, score in per_word.items():
                best, count, total = scores.get(idx, (0.0, 0, 0.0))
                scores[idx] = (max(best, score), count + 1, total + score)
        return scores

    def _clarify_ids(self, ranked) -> List[int]:
        best = ranked[0][1][0]
        return [idx for idx, (s, _, _) in ranked if best - s < _SCORE_MARGIN][:5]

    def _clarification(self, ids: List[int]) -> str:
        options = [
            f"- {self._employees[i]['full_name']} (отдел «{self._employees[i].get('department', '')}»)"
            for i in ids
        ]
        listing = "\n".join(options)
        return f"{CLARIFICATION_PREFIX}\nУточните, пожалуйста:\n{listing}"

    @staticmethod
    def _asks_manager(query: str) -> bool:
        lowered = query.lower()
        return any(word in lowered for word in _MANAGER_WORDS)

    @staticmethod
    def _card(emp: dict) -> str:
        return (
            f"**{emp.get('full_name', '')}**\n"
            f"Должность: {emp.get('position', '')}\n"
            f"Отдел: {emp.get('department', '')}\n"
            f"Департамент: {emp.get('division', '')}\n"
            f"Внутренний номер: {emp.get('phone', '')}\n"
            f"Почта: {emp.get('email', '')}"
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
