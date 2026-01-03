from __future__ import annotations
import json
from pathlib import Path

from agent.state import AgentState
from agent.graph import build_graph


DATA_DIR = Path(__file__).resolve().parent / "data"


def load(name: str):
    return json.loads((DATA_DIR / name).read_text(encoding="utf-8"))


def main():
    customers = load("customers.json")
    customer = customers[0]

    state = AgentState(
        user_profile=customer["profile"],
        user_logs=customer["logs"],
        context=customer["context"],
        max_retries=1,
    )

    app = build_graph()
    out: AgentState = app.invoke(state)

    result = {
        "persona": out.persona,
        "persona_reason": out.persona_reason,
        "candidates": [{"name": p["name"], "similarity": p.get("similarity")} for p in out.candidate_products],
        "selected_product": out.selected_product,
        "messages": out.generated_messages,
        "retry_count": out.retry_count,
        "feedback": out.feedback,
        "is_valid": out.is_valid,
    }

    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()