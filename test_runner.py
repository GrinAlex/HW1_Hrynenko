import json
import time
from pydantic import ValidationError

from agent import run_agent
from logger import TrajectoryLogger
from safety import LoopDetector, is_timeout_exceeded
from tools import CreatePromoInput, PROMO_DB, create_promo, deactivate_promo, search_promos


def run_unit_and_safety_checks() -> None:
    """Демонстрація та перевірка валідації tools і захисних механізмів (Завдання 1 та 3)."""
    print("=== 1. Unit-тести інструментів та Pydantic-валідації ===")

    # Успішний пошук
    res_search = json.loads(search_promos.invoke({"category": "electronics", "only_active": True}))
    assert res_search["status"] == "success" and res_search["count"] >= 1
    print("[OK] search_promos працює коректно.")

    # Некоректна знижка (> 70%) має відхилятися валідатором
    try:
        CreatePromoInput(
            promo_code="TOO_HIGH",
            category="home",
            discount_percent=95,
            start_date="2026-12-01",
            end_date="2026-12-10",
        )
        raise AssertionError("Очікувалась помилка валідації!")
    except ValidationError:
        print("[OK] Валідація discount_percent > 70 працює.")

    # Некоректний порядок дат
    try:
        CreatePromoInput(
            promo_code="BAD_DATES",
            category="home",
            discount_percent=20,
            start_date="2026-12-15",
            end_date="2026-12-01",
        )
        raise AssertionError("Очікувалась помилка валідації дат!")
    except ValidationError:
        print("[OK] Крос-валідація дат (end_date <= start_date) працює.")

    print("\n=== 2. Перевірка захисних механізмів (LoopDetector, timeout) ===")
    detector = LoopDetector(max_repeats=3)
    assert detector.check("search_promos", {"category": "all"}) is False
    assert detector.check("search_promos", {"category": "all"}) is False
    assert detector.check("search_promos", {"category": "all"}) is True
    print("[OK] LoopDetector виявив зациклення на 3-му повторі.")

    past_start = time.monotonic() - 10
    assert is_timeout_exceeded(past_start, timeout_seconds=5) is True
    print("[OK] Перевірка timeout спрацювала коректно.\n")


TEST_CASES = [
    {
        "id": "TC-001",
        "complexity": "simple",
        "query": "Покажи всі активні акції у категорії electronics.",
        "expected": "Список активних акцій для electronics (включно з TECH_START).",
    },
    {
        "id": "TC-002",
        "complexity": "medium",
        "query": (
            "Перевір активні акції на електроніку та створи нову акцію BLACK_FRIDAY_26 "
            "для категорії electronics зі знижкою 25% з 2026-11-25 по 2026-11-30."
        ),
        "expected": "Виклик search_promos, потім створення BLACK_FRIDAY_26 зі знижкою 25%.",
    },
    {
        "id": "TC-003",
        "complexity": "medium",
        "query": (
            "Створи акцію MEGA_DROP для категорії home зі знижкою 85% "
            "(або максимально дозволеною за правилами) з 2026-12-01 по 2026-12-10."
        ),
        "expected": "Агент дотримується ліміту 70% або виправляє помилку валідації та створює акцію зі знижкою 70%.",
    },
    {
        "id": "TC-004",
        "complexity": "complex",
        "query": (
            "Знайди активні акції для категорії home, деактивуй акцію HOME_COZY "
            "з причиною 'Завершення осінньої кампанії' і створи нову акцію WINTER_HOME "
            "зі знижкою 30% з 2026-12-01 по 2026-12-25."
        ),
        "expected": "Послідовне виконання search_promos → deactivate_promo(HOME_COZY) → create_promo(WINTER_HOME).",
    },
]


def run_e2e_tests() -> None:
    """Проганяє 4 тест-кейси через LangGraph-агента та зберігає trajectory.json і test_results.json."""
    print("=== 3. Запуск E2E тест-кейсів на LangGraph-агенті ===")
    results = []
    combined_logger = TrajectoryLogger()

    for tc in TEST_CASES:
        print(f"Running {tc['id']} ({tc['complexity']}): {tc['query'][:60]}...")
        start = time.monotonic()
        try:
            # Останній тест-кейс збереже детальну траєкторію у trajectory.json
            state = run_agent(tc["query"], save_trajectory_path="trajectory.json")
            structured = state.get("structured_response") or {}
            actual = structured.get("answer") or str(state["messages"][-1].content)
            steps = state.get("step_count", 0)
            stop_reason = state.get("stop_reason")
            status = "success" if not stop_reason else f"stopped ({stop_reason})"

            tool_calls = [
                tc_call["name"]
                for msg in state.get("messages", [])
                if hasattr(msg, "tool_calls") and msg.tool_calls
                for tc_call in msg.tool_calls
            ]
        except Exception as exc:
            actual = f"Error: {exc}"
            steps = -1
            status = "error"
            tool_calls = []

        elapsed_ms = int((time.monotonic() - start) * 1000)
        results.append(
            {
                "test_id": tc["id"],
                "complexity": tc["complexity"],
                "query": tc["query"],
                "expected": tc["expected"],
                "actual": actual[:500],
                "status": status,
                "steps": steps,
                "tool_calls": tool_calls,
                "elapsed_ms": elapsed_ms,
            }
        )
        print(f"  -> {tc['id']}: {status} | steps={steps} | tools={tool_calls} | {elapsed_ms} ms")

    with open("test_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\n[OK] Збережено результати {len(results)} тест-кейсів у test_results.json та траєкторію у trajectory.json.")


if __name__ == "__main__":
    run_unit_and_safety_checks()
    run_e2e_tests()