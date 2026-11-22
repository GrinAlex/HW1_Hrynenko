import operator
import os
import time
from typing import Annotated, Any, Literal, TypedDict

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode
from pydantic import BaseModel, Field

from logger import TrajectoryLogger
from safety import MAX_STEPS, TIMEOUT_SECONDS, LoopDetector, is_timeout_exceeded
from tools import TOOLS_LIST

load_dotenv()

SYSTEM_PROMPT = (
    "Ти надійний AI-асистент відділу маркетингу для управління ціновими акціями та промокодами.\n"
    "Правила:\n"
    "1. Перед створенням нової акції завжди перевіряй наявні акції через `search_promos`.\n"
    "2. Дотримуйся схем інструментів (знижка 1–70%, промокод у форматі A-Z, 0-9, _).\n"
    "3. Якщо інструмент повернув помилку валідації (наприклад, знижка понад 70%), "
    "скоригуй параметри або поясни обмеження користувачу.\n"
    "4. Відповідай чітко українською мовою."
)


# ── Structured Output модель ────────────────────────────────────────────────

class PromoAgentResponse(BaseModel):
    """Структурована фінальна відповідь агента."""

    answer: str = Field(description="Чітка фінальна відповідь користувачу українською мовою.")
    confidence: float = Field(ge=0.0, le=1.0, description="Рівень впевненості у відповіді (від 0.0 до 1.0).")
    sources: list[str] = Field(
        default_factory=list,
        description="Список використаних інструментів або промокодів, на які спирається відповідь.",
    )


# ── Стан графа ──────────────────────────────────────────────────────────────

class AgentState(TypedDict):
    messages: Annotated[list[Any], operator.add]
    step_count: int
    start_time: float
    stop_reason: str | None
    structured_response: dict[str, Any] | None


def create_llm() -> ChatOpenAI:
    """Створює клієнт LLM (підтримує локальну Ollama та OpenRouter через .env)."""
    return ChatOpenAI(
        base_url=os.getenv("LLM_BASE_URL", "http://localhost:11434/v1"),
        api_key=os.getenv("LLM_API_KEY", "ollama"),
        model=os.getenv("LLM_MODEL", "llama3.1:8b"),
        temperature=0.1,
    )


def build_agent_graph(
    llm: ChatOpenAI | None = None,
    trajectory_logger: TrajectoryLogger | None = None,
    max_steps: int = MAX_STEPS,
    timeout_seconds: float = TIMEOUT_SECONDS,
    max_repeats: int = 3,
):
    """Збирає та компілює LangGraph StateGraph із захисними механізмами."""
    llm = llm or create_llm()
    llm_with_tools = llm.bind_tools(TOOLS_LIST)
    structured_llm = llm.with_structured_output(PromoAgentResponse)
    tool_node = ToolNode(TOOLS_LIST, handle_tool_errors=True)
    loop_detector = LoopDetector(max_repeats=max_repeats)
    t_logger = trajectory_logger or TrajectoryLogger()

    def agent_node(state: AgentState) -> dict[str, Any]:
        step = state.get("step_count", 0)
        start_time = state.get("start_time", time.monotonic())

        # 1. Перевірка timeout
        if is_timeout_exceeded(start_time, timeout_seconds):
            msg = AIMessage(content=f"Перевищено тайм-аут виконання ({timeout_seconds} сек). Роботу зупинено.")
            t_logger.log_step(step + 1, "agent", "timeout_check", msg.content)
            return {"messages": [msg], "step_count": step, "stop_reason": "timeout"}

        # 2. Перевірка max_steps
        if step >= max_steps:
            msg = AIMessage(content=f"Досягнуто ліміт кроків ({max_steps}). Часткова відповідь сформована.")
            t_logger.log_step(step + 1, "agent", "max_steps_check", msg.content)
            return {"messages": [msg], "step_count": step, "stop_reason": "max_steps"}

        # 3. Виклик LLM
        response = llm_with_tools.invoke(state["messages"])
        tool_calls = getattr(response, "tool_calls", []) or []

        # 4. Перевірка зациклення (LoopDetector)
        for tc in tool_calls:
            if loop_detector.check(tc["name"], tc.get("args", {})):
                msg = AIMessage(
                    content=f"Виявлено зациклення на виклику інструменту '{tc['name']}'. Виконання зупинено."
                )
                t_logger.log_step(step + 1, "agent", "loop_check", msg.content, [tc["name"]])
                return {"messages": [msg], "step_count": step + 1, "stop_reason": "loop_detected"}

        last_in = state["messages"][-1].content if state["messages"] else ""
        t_logger.log_step(
            step + 1,
            "agent",
            str(last_in),
            str(response.content or f"tool_calls: {[tc['name'] for tc in tool_calls]}"),
            [tc["name"] for tc in tool_calls],
        )
        return {"messages": [response], "step_count": step + 1, "stop_reason": None}

    def tools_wrapper_node(state: AgentState) -> dict[str, Any]:
        result = tool_node.invoke(state)
        tool_msgs = result.get("messages", [])
        outputs = " | ".join(str(m.content) for m in tool_msgs)
        t_logger.log_step(state.get("step_count", 0), "tools", "execute_tools", outputs)
        return result

    def formatter_node(state: AgentState) -> dict[str, Any]:
        """Конвертує фінальну відповідь у структурований Pydantic-формат."""
        last_text = str(state["messages"][-1].content)
        used_tools = sorted(
            {
                tc["name"]
                for m in state["messages"]
                if hasattr(m, "tool_calls") and m.tool_calls
                for tc in m.tool_calls
            }
        )
        try:
            structured: PromoAgentResponse = structured_llm.invoke(
                [
                    SystemMessage(
                        content="Сформуй структуровану відповідь на основі тексту нижче. Мова — українська."
                    ),
                    HumanMessage(content=f"Текст відповіді: {last_text}\nВикористані інструменти: {used_tools}"),
                ]
            )
            data = structured.model_dump()
        except Exception:
            # Безпечний fallback для локальних 8B моделей, якщо JSON-форматування дало збій
            data = PromoAgentResponse(
                answer=last_text,
                confidence=0.8 if not state.get("stop_reason") else 0.4,
                sources=used_tools,
            ).model_dump()

        t_logger.log_step(state.get("step_count", 0), "formatter", last_text, str(data))
        return {"structured_response": data}

    def should_continue(state: AgentState) -> Literal["tools", "formatter"]:
        if state.get("stop_reason"):
            return "formatter"
        last_msg = state["messages"][-1]
        if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
            return "tools"
        return "formatter"

    graph = StateGraph(AgentState)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", tools_wrapper_node)
    graph.add_node("formatter", formatter_node)

    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", should_continue)
    graph.add_edge("tools", "agent")
    graph.add_edge("formatter", END)

    return graph.compile()


def run_agent(
    query: str,
    max_steps: int = MAX_STEPS,
    timeout_seconds: float = TIMEOUT_SECONDS,
    save_trajectory_path: str | None = "trajectory.json",
) -> dict[str, Any]:
    """Запускає LangGraph-агента на одному запиті та повертає повний стан."""
    t_logger = TrajectoryLogger()
    app = build_agent_graph(
        trajectory_logger=t_logger,
        max_steps=max_steps,
        timeout_seconds=timeout_seconds,
    )
    result = app.invoke(
        {
            "messages": [SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=query)],
            "step_count": 0,
            "start_time": time.monotonic(),
            "stop_reason": None,
            "structured_response": None,
        }
    )
    if save_trajectory_path:
        t_logger.save(save_trajectory_path)
    return result


if __name__ == "__main__":
    res = run_agent("Перевір активні акції на електроніку.")
    print("Структурована відповідь:", res["structured_response"])