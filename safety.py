import hashlib
import json
import time

MAX_STEPS = 10
TIMEOUT_SECONDS = 90


def is_timeout_exceeded(start_time: float, timeout_seconds: float = TIMEOUT_SECONDS) -> bool:
    """Перевіряє, чи перевищено загальний ліміт часу виконання агента."""
    return (time.monotonic() - start_time) >= timeout_seconds


class LoopDetector:
    """Детектор зациклення агента — відстежує повторювані виклики одного й того самого tool."""

    def __init__(self, max_repeats: int = 3) -> None:
        self.max_repeats = max_repeats
        self.recent_calls: list[str] = []

    def check(self, tool_name: str, args: dict) -> bool:
        """Повертає True, якщо той самий tool з тими самими аргументами викликано max_repeats разів поспіль."""
        serialized_args = json.dumps(args, sort_keys=True, ensure_ascii=False, default=str)
        call_hash = hashlib.md5(f"{tool_name}:{serialized_args}".encode("utf-8")).hexdigest()
        self.recent_calls.append(call_hash)

        if len(self.recent_calls) >= self.max_repeats:
            last_n = self.recent_calls[-self.max_repeats :]
            if len(set(last_n)) == 1:
                return True
        return False

    def reset(self) -> None:
        """Очищає історію викликів перед новим запуском."""
        self.recent_calls.clear()