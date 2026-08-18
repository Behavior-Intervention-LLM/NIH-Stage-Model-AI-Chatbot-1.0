"""

 Pydantic 
"""
from typing import Dict, List, Optional, Literal, Any
from pydantic import BaseModel, Field
from datetime import datetime
from enum import Enum


# ====================  ====================

class MessageRole(str, Enum):
    """"""
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


class Message(BaseModel):
    """"""
    role: MessageRole
    content: str
    timestamp: datetime = Field(default_factory=datetime.now)


# ==================== Slots（） ====================

class StageSlots(BaseModel):
    """Stage """
    need_stage: Optional[bool] = None
    # Canonical keys from app/core/stage_model.py:
    # "0", "IA", "IB", "II", "III", "IV", "V" (bare "I" when Stage I's two
    # sub-stages cannot be distinguished).
    stage: Optional[str] = None
    stage_confidence: float = 0.0
    
    user_goal: Optional[str] = None
    
    # Stage 
    intervention_defined: Optional[bool] = None
    manualized: Optional[bool] = None
    mechanism_tested: Optional[bool] = None
    efficacy_tested: Optional[bool] = None
    effectiveness_tested: Optional[bool] = None
    implementation_tested: Optional[bool] = None
    
    # （）
    extracted_features: Dict[str, Any] = Field(default_factory=dict)


# ==================== Artifacts（） ====================

class Citation(BaseModel):
    """Reference"""
    source: str
    passage: str
    relevance_score: Optional[float] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class Artifact(BaseModel):
    """"""
    tool_name: str
    result_type: Literal["text", "structured", "citations", "raw"]
    content: Any
    citations: List[Citation] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)


# ====================  ====================

class Attachment(BaseModel):
    """One file the user attached to this conversation.

    Attachments are working context for the conversation, not corpus
    material: they are never chunked into the vector store, they live only in
    memory, and they disappear with the session.
    """
    name: str
    text: str
    added_at: datetime = Field(default_factory=datetime.now)

    @property
    def chars(self) -> int:
        return len(self.text)


class SessionState(BaseModel):
    """"""
    session_id: str
    messages: List[Message] = Field(default_factory=list)  #  N （）
    summary: Optional[str] = None  # （）
    slots: StageSlots = Field(default_factory=StageSlots) # If you ask about the stage, reminder of what stage is in
    artifacts: List[Artifact] = Field(default_factory=list)  # tool
    # Deliberately a sibling of `slots`, not a field inside it: the responder
    # serialises slots into a length-capped block, and attachment text placed
    # there consumed ~89% of that budget and pushed the stage/workflow fields
    # it exists to carry off the end.
    attachments: List[Attachment] = Field(default_factory=list)
    last_route: Optional[str] = None  #  router （ debug）
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)

    def add_message(self, role: MessageRole, content: str):
        """"""
        self.messages.append(Message(role=role, content=content))
        self.updated_at = datetime.now()

    def get_recent_messages(self, n: int = 20) -> List[Message]:
        """ N """
        return self.messages[-n:]

    def add_attachment(self, name: str, text: str) -> bool:
        """Attach a file's text. Re-attaching the same content is a no-op."""
        text = (text or "").strip()
        if not text:
            return False
        for existing in self.attachments:
            if existing.name == name and existing.text == text:
                return False
        self.attachments.append(Attachment(name=name, text=text))
        self.updated_at = datetime.now()
        return True

    def clear_attachments(self) -> int:
        removed = len(self.attachments)
        self.attachments = []
        self.updated_at = datetime.now()
        return removed

    def attachment_context(self, budget: int) -> str:
        """Render attachments for a prompt, sharing `budget` chars between them.

        Every attachment gets a share rather than the first one taking the
        whole budget, so attaching a second document cannot silently drop it.
        Truncation is announced in-band so the model knows it is seeing part
        of a document and can say so.
        """
        if not self.attachments or budget <= 0:
            return ""

        # Headers and truncation notices are part of what gets sent, so they
        # come out of the budget rather than being added on top of it.
        overhead = sum(len(f"--- Attachment: {a.name} ---\n\n\n") + 60 for a in self.attachments)
        usable = max(len(self.attachments) * 200, budget - overhead)
        per_file = usable // len(self.attachments)

        blocks = []
        for att in self.attachments:
            body = att.text
            if len(body) > per_file:
                body = (
                    body[:per_file]
                    + f"\n… [truncated: showing {per_file} of {len(att.text)} characters]"
                )
            blocks.append(f"--- Attachment: {att.name} ---\n{body}")
        return "\n\n".join(blocks)


# ==================== Agent output ====================

class ToolCall(BaseModel):
    """"""
    tool_name: str
    tool_args: Dict[str, Any]
    success_criteria: Optional[str] = None
    # Tool/runtime result (e.g. RAG chunks); optional until after execution
    output: Optional[Any] = None


class AgentOutput(BaseModel):
    """Agent criteriaoutput（）"""
    decision: Dict[str, Any] = Field(default_factory=dict)  # （ need_stage, stage）
    confidence: float = 0.0  # 0~1
    analysis: str = ""  # （ agent ，）
    actions: List[ToolCall] = Field(default_factory=list)  # recommendationnext step tool call
    user_facing: Optional[str] = None  #  router Direct reply，
    metadata: Dict[str, Any] = Field(default_factory=dict)


# ==================== Router output ====================

class RouteMode(str, Enum):
    """Route mode"""
    DIRECT_REPLY = "direct_reply"  # Direct reply
    NEED_CLARIFY = "need_clarify"  # 
    STAGE_FLOW = "stage_flow"  # Stage 
    NON_STAGE_TASK = "non_stage_task"  #  Stage 


class RoutePlan(BaseModel):
    """Router output"""
    calls: List[str] = Field(default_factory=list)  #  agent （）
    mode: RouteMode = RouteMode.DIRECT_REPLY
    notes: str = ""  # Debug info


# ==================== Planner output ====================

class PlanStepType(str, Enum):
    """"""
    ASK_USER = "ask_user"  # 
    CALL_TOOL = "call_tool"  # 
    DRAFT_OUTPUT = "draft_output"  # output
    VERIFY = "verify"  # 


class PlanStep(BaseModel):
    """"""
    step_type: PlanStepType
    tool_name: Optional[str] = None  #  call_tool
    tool_args_schema: Optional[Dict[str, Any]] = None
    success_criteria: Optional[str] = None
    description: str = ""


class PlannerOutput(BaseModel):
    """Planner Agent output"""
    plan_steps: List[PlanStep] = Field(default_factory=list)
    next_question: Optional[str] = None  # 
    final_response_outline: Optional[str] = None  # Responder 


# ==================== Tool  ====================

class ToolResult(BaseModel):
    """"""
    text: Optional[str] = None
    structured: Optional[Dict[str, Any]] = None
    citations: List[Citation] = Field(default_factory=list)
    raw: Optional[Any] = None
    success: bool = True
    error: Optional[str] = None


# ==================== API / ====================

# Intent section not done through intent agent
class ChatRequest(BaseModel):
    """Chat request"""
    session_id: Optional[str] = None
    message: str
    workflow: Optional[Literal["auto", "navigator", "mechanism_coach", "study_builder", "measure_finder", "grant_partner"]] = None # The intent is here, do we need intent agent
    # Pre-extracted text of a file the caller is attaching to the conversation.
    # Working context only: never indexed into the vector store, never
    # persisted, discarded with the session. The server does no file parsing —
    # API callers extract text themselves; the chat UI does it client-side.
    document_text: Optional[str] = None
    document_name: Optional[str] = None


class ChatResponse(BaseModel):
    """Chat response"""
    session_id: str
    reply: str
    debug: Dict[str, Any] = Field(default_factory=dict)  # Debug info
    citations: List[Citation] = Field(default_factory=list)  # Reference
    next_question: Optional[str] = None  #
    # Quote this to POST /feedback/rating. None when feedback is disabled or
    # the turn never reached the orchestrator (e.g. a guardrail rejection).
    turn_uid: Optional[str] = None


class RatingRequest(BaseModel):
    """Explicit thumbs up/down on one answer, with an optional comment."""
    turn_uid: str
    # +1 thumbs up, -1 thumbs down, null to withdraw a previous rating.
    rating: Optional[Literal[-1, 1]] = None
    comment: Optional[str] = None
