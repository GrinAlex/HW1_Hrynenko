from datetime import datetime, timezone
import json
import time
from typing import Any


class TrajectoryLogger:
    """Логер траєкторії виконання ReAct-агента у форматі JSON."""

    def __init__(self) -> None:
        self.steps: list[dict[str, Any]] = []
        self.start_time: float = time.monotonic()

    def reset(self) -> None:
        self.steps = []
        self.start_time = time.monotonic()

    def log_step(
        self,
        step_num: int,
        node: str,
        input_data: str,
        output_data: str,
        tool_calls: list[str] | None = None,
    ) -> None:
        """Додає запис про крок виконання вузла графа."""
        self.steps.append(
            {
                "step": step_num,
                "node": node,
                "input": str(input_data)[:500],
                "output": str(output_data)[:500],
                "tool_calls": tool_calls or [],
                "timestamp": datetime.now(tz=timezone.utc).isoformat(),
                "elapsed_ms": int((time.monotonic() - self.start_time) * 1000),
            }
        )

    def save(self, filepath: str = "trajectory.json") -> None:
        """Зберігає повну траєкторію у JSON-файл."""
        payload = {
            "total_steps": len(self.steps),
            "total_time_ms": int((time.monotonic() - self.start_time) * 1000),
            "trajectory": self.steps,
        }
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)