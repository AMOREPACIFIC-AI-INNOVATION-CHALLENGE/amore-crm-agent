from __future__ import annotations
from typing import Literal

from langgraph.graph import StateGraph, END
from agent.state import AgentState
from agent.nodes import (
    node1_persona_classifier,
    node2_candidate_retrieval,
    node3_recommender,
    node4_prompt_builder,
    node5_copywriter_llm,
    node6_compliance_guardian,
    node7_final_output,
)

def route_after_guardian(state: AgentState) -> Literal["retry", "final"]:
    if state.get("is_valid", False):
        return "final"
    if int(state.get("retry_count", 0)) >= int(state.get("max_retries", 1)):
        return "final"
    return "retry"

def build_graph():
    g = StateGraph(AgentState)

    g.add_node("persona", node1_persona_classifier)
    g.add_node("retrieve", node2_candidate_retrieval)
    g.add_node("recommend", node3_recommender)
    g.add_node("prompt", node4_prompt_builder)
    g.add_node("copy", node5_copywriter_llm)
    g.add_node("guard", node6_compliance_guardian)
    g.add_node("final", node7_final_output)

    g.set_entry_point("persona")
    g.add_edge("persona", "retrieve")
    g.add_edge("retrieve", "recommend")
    g.add_edge("recommend", "prompt")
    g.add_edge("prompt", "copy")
    g.add_edge("copy", "guard")

    g.add_conditional_edges(
        "guard",
        route_after_guardian,
        {"retry": "copy", "final": "final"},
    )

    g.add_edge("final", END)
    return g.compile()