"""Память диалогов бота: последние N реплик по каждому пользователю."""

from collections import deque
from typing import Dict, List, Tuple


class SessionMemory:
    """
    Хранит историю диалога в памяти процесса: user_id → deque(maxlen=max_turns).
    Формат — кросс-задачный контракт с RAGPipeline: список пар (role, text),
    role ∈ {"user", "assistant"}.
    """

    def __init__(self, max_turns: int = 6):
        self._store: Dict[str, deque] = {}
        self._max_turns = max_turns

    def history(self, user_id: str) -> List[Tuple[str, str]]:
        """Последние max_turns реплик пользователя, старые вытесняются."""
        return list(self._store.get(user_id, ()))

    def append(self, user_id: str, role: str, text: str) -> None:
        if user_id not in self._store:
            self._store[user_id] = deque(maxlen=self._max_turns)
        self._store[user_id].append((role, text))
