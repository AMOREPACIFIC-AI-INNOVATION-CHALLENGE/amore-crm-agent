from __future__ import annotations
from typing import Any, Dict, List, Optional, TypedDict

class AgentState(TypedDict, total=False):
    # input
    user_profile: Dict[str, Any]
    user_logs: Dict[str, Any]
    context: Dict[str, Any]

    # node outputs
    persona: str
    persona_reason: str
    candidate_products: List[Dict[str, Any]]
    selected_product: Dict[str, Any]

    fact_data: Dict[str, Any]
    system_prompt: str

    generated_messages: List[Dict[str, Any]]
    feedback: str

    # loop control
    retry_count: int
    max_retries: int
    is_valid: bool

    # final
    final_output: Dict[str, Any]