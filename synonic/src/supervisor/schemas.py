"""
Structured decision the supervisor LLM must emit each turn. The supervisor
never returns free text like "I think we should search more" - it returns
one of these, validated, and executor.execute_decision is the only code
allowed to turn it into a tool call.
"""

from typing import Literal

from pydantic import BaseModel, Field

ActionName = Literal["retrieve", "generate", "merge", "score", "finalize", "finish"]

ALLOWED_ACTIONS: set[str] = {"retrieve", "generate", "merge", "score", "finalize", "finish"}


class SupervisorDecision(BaseModel):
    action: ActionName
    reason: str = Field(description="Short, verifiable justification for the chosen action.")
    parameters: dict = Field(default_factory=dict)
    confidence: float = Field(ge=0.0, le=1.0, default=0.0)
