"""
Intent Agent: identify user intent and extract structured request signals.
"""
import re
from typing import Literal

from app.agents.base import BaseAgent
from app.core.llm import llm_client
from app.core.types import AgentOutput, SessionState


class IntentAgent(BaseAgent):
    """Intent recognition and request extraction agent."""

    INTENT_LABELS = Literal[
        "stage_guidance",
        "general_qa",
        "chit_chat",
        "admin",
        "debug",
        "unknown",
    ]

    def __init__(self):
        super().__init__("IntentAgent")

    @staticmethod
    def _infer_workflow_from_text(message_lower: str) -> str:
        if any(k in message_lower for k in ["grant", "specific aims", "aim 1", "aim 2", "reviewer", "revision plan"]):
            return "grant_partner"
        if any(k in message_lower for k in ["measure", "scale", "instrument", "psychometric", "questionnaire"]):
            return "measure_finder"
        if any(k in message_lower for k in ["study design", "design matrix", "comparator", "fidelity", "sample size"]):
            return "study_builder"
        if any(k in message_lower for k in ["mechanism", "mediator", "pathway", "manipulat", "verify mechanism"]):
            return "mechanism_coach"
        return "navigator"

    @staticmethod
    def _detect_language(text: str) -> str:
        return "zh" if re.search(r"[\u4e00-\u9fff]", text or "") else "en"

    def run(self, state: SessionState, user_message: str, context: str = "") -> AgentOutput:
        llm_output = self._run_with_llm(user_message, context)
        if llm_output:
            return llm_output
        return self._run_with_rules(user_message)

    def _run_with_llm(self, user_message: str, context: str = "") -> AgentOutput | None:
        if not llm_client.is_enabled():
            return None

        system_prompt = (
            "You are an intent and information extraction agent. Output JSON only.\n"
            "Fields:\n"
            "- workflow (navigator/mechanism_coach/study_builder/measure_finder/grant_partner)\n"
            "- need_stage (bool)\n"
            "- intent_label (stage_guidance/general_qa/chit_chat/admin/debug/unknown)\n"
            "- query_type (definition/stage_classification/stage_requirements/next_step/general_qa/chit_chat/admin)\n"
            "- language (zh/en)\n"
            "- is_definition_query (bool)\n"
            "- confidence (0~1)\n"
            "- user_goal (string|null)\n"
            "- extracted_signals (list[string])\n"
            "- missing_info (list[string])\n"
            "- clarifying_question (string|null)"
        )
        user_prompt = (
            f"user_message: {user_message}\n"
            f"context: {context[:1200]}\n"
            "Classify and extract fields for downstream responder."
        )
        data = llm_client.chat_json(system_prompt=system_prompt, user_prompt=user_prompt)
        if not data:
            return None

        workflow = str(data.get("workflow", "")).lower().strip()
        valid_workflows = {"navigator", "mechanism_coach", "study_builder", "measure_finder", "grant_partner"}
        if workflow not in valid_workflows:
            workflow = self._infer_workflow_from_text(user_message.lower())

        intent_label = str(data.get("intent_label", "unknown"))
        valid_labels = {"stage_guidance", "general_qa", "chit_chat", "admin", "debug", "unknown"}
        if intent_label not in valid_labels:
            intent_label = "unknown"

        confidence = max(0.0, min(1.0, float(data.get("confidence", 0.6))))
        language = str(data.get("language", self._detect_language(user_message))).lower()
        if language not in {"zh", "en"}:
            language = self._detect_language(user_message)

        query_type = str(data.get("query_type", "general_qa")).lower()
        valid_query_types = {
            "definition",
            "stage_classification",
            "stage_requirements",
            "next_step",
            "general_qa",
            "chit_chat",
            "admin",
        }
        if query_type not in valid_query_types:
            query_type = "general_qa"

        is_definition_query = bool(data.get("is_definition_query", query_type == "definition"))
        need_stage = bool(data.get("need_stage", intent_label == "stage_guidance" and not is_definition_query))

        # Consistency correction: if user intent is stage-related and not a definition query,
        # enforce stage flow gate even when raw LLM booleans are noisy.
        if query_type in {"stage_classification", "stage_requirements", "next_step"} and not is_definition_query:
            need_stage = True
        if query_type == "definition" or is_definition_query:
            need_stage = False
        if workflow in {"mechanism_coach", "study_builder", "measure_finder", "grant_partner"} and query_type not in {"definition", "chit_chat", "admin"}:
            need_stage = True

        user_goal = data.get("user_goal")
        if user_goal is not None:
            user_goal = str(user_goal).strip() or None

        extracted_signals = data.get("extracted_signals", []) or []
        if not isinstance(extracted_signals, list):
            extracted_signals = []
        extracted_signals = [str(x) for x in extracted_signals[:6]]

        missing_info = data.get("missing_info", []) or []
        if not isinstance(missing_info, list):
            missing_info = []
        missing_info = [str(x) for x in missing_info[:5]]

        clarifying = data.get("clarifying_question")
        if clarifying is not None:
            clarifying = str(clarifying).strip() or None
        if confidence >= 0.6:
            clarifying = None

        decision = {
            "workflow": workflow,
            "need_stage": need_stage,
            "intent_label": intent_label,
            "query_type": query_type,
            "language": language,
            "is_definition_query": is_definition_query,
            "user_goal": user_goal,
            "extracted_signals": extracted_signals,
            "missing_info": missing_info,
            "clarifying_question": clarifying,
        }
        return AgentOutput(
            decision=decision,
            confidence=confidence,
            analysis=f"LLM workflow={workflow}, intent={intent_label}, query_type={query_type}, need_stage={need_stage}",
            user_facing=clarifying,
        )

    def _run_with_rules(self, user_message: str) -> AgentOutput:
        message_lower = user_message.lower()
        language = self._detect_language(user_message)
        workflow = self._infer_workflow_from_text(message_lower)

        need_stage = False
        intent_label = "unknown"
        confidence = 0.45
        query_type = "general_qa"

        stage_keywords = [
            "nih stage",
            "stage model",
            "stage 0",
            "stage i",
            "stage ii",
            "stage iii",
            "stage iv",
            "stage v",
            "intervention",
            "mechanism",
            "efficacy",
            "effectiveness",
            "implementation",
            "sustainability",
            "stage",
            "intervention",
            "mechanism",
            "efficacy",
            "effectiveness",
            "implementation",
        ]
        stage_task_keywords = ["requirements", "criteria", "next step", "what should", "requirements", "next step", "how to proceed", "recommendation"]
        qa_keywords = ["what is", "what are", "explain", "tell me", "define", "what is", "explain", "introduce"]
        chit_chat_keywords = ["hello", "hi", "thanks", "thank you", "bye", "how are you", "hello", "hi", "thanks"]

        is_definition_query = (
            any(
                k in message_lower
                for k in [
                    "what is",
                    "what's",
                    "define",
                    "explain",
                    "introduce",
                    "how many stages",
                    "number of stages",
                    "list stages",
                ]
            )
            and any(k in message_lower for k in ["nih stage model", "nih stage", "stage model", "stage model"])
        )

        if any(k in message_lower for k in ["next step", "what should", "next step", "how to proceed", "recommendation"]):
            query_type = "next_step"
        elif any(k in message_lower for k in ["requirement", "requirements", "criteria", "requirements", "criteria"]):
            query_type = "stage_requirements"
        elif is_definition_query:
            query_type = "definition"
        elif "stage" in message_lower or "stage" in message_lower:
            query_type = "stage_classification"

        stage_score = sum(1 for kw in stage_keywords if kw in message_lower)
        stage_task_score = sum(1 for kw in stage_task_keywords if kw in message_lower)
        qa_score = sum(1 for kw in qa_keywords if kw in message_lower)
        chat_score = sum(1 for kw in chit_chat_keywords if kw in message_lower)

        if message_lower.startswith("/"):
            intent_label = "admin"
            confidence = 0.9
            query_type = "admin"
            workflow = "navigator"
        elif chat_score > 0 and len(message_lower.split()) <= 8:
            intent_label = "chit_chat"
            confidence = 0.85
            query_type = "chit_chat"
            workflow = "navigator"
        elif is_definition_query:
            intent_label = "general_qa"
            need_stage = False
            confidence = 0.88
            query_type = "definition"
        elif stage_score + stage_task_score >= 1:
            intent_label = "stage_guidance"
            need_stage = True
            confidence = min(0.9, 0.62 + 0.08 * (stage_score + stage_task_score))
        elif qa_score > 0:
            intent_label = "general_qa"
            confidence = min(0.8, 0.62 + 0.08 * qa_score)
        else:
            intent_label = "general_qa"
            confidence = 0.55

        if workflow in {"mechanism_coach", "study_builder", "measure_finder", "grant_partner"} and query_type not in {"definition", "chit_chat", "admin"}:
            need_stage = True

        extracted_signals = []
        if is_definition_query:
            extracted_signals.append("definition_query")
        if "pilot" in message_lower or "feasibility" in message_lower:
            extracted_signals.append("feasibility_signal")
        if "rct" in message_lower or "randomized" in message_lower:
            extracted_signals.append("rct_signal")
        if "mechanism" in message_lower or "mechanism" in message_lower:
            extracted_signals.append("mechanism_signal")

        user_goal = None
        if query_type == "next_step":
            user_goal = "Get actionable next-step guidance"

        missing_info = []
        if need_stage and query_type in {"stage_classification", "next_step"}:
            if language == "zh":
                missing_info = ["（pilot  RCT）", "", "efficacy/effectiveness"]
            else:
                missing_info = ["study design (pilot vs RCT)", "sample size", "availability of efficacy/effectiveness outcomes"]

        clarifying_question = None
        if confidence < 0.6:
            clarifying_question = (
                " NIH Stage Model recommendation，general QA？"
                if language == "zh"
                else "Are you asking for NIH Stage Model stage guidance or general QA?"
            )
        elif need_stage and query_type in {"stage_classification", "next_step"} and confidence < 0.75:
            clarifying_question = (
                "To improve stage accuracy, please share study design, sample size, and key outcomes."
                if language == "zh"
                else "To improve stage accuracy, please share study design, sample size, and key outcomes."
            )

        decision = {
            "workflow": workflow,
            "need_stage": need_stage,
            "intent_label": intent_label,
            "query_type": query_type,
            "language": language,
            "is_definition_query": is_definition_query,
            "user_goal": user_goal,
            "extracted_signals": extracted_signals,
            "missing_info": missing_info,
            "clarifying_question": clarifying_question,
        }
        return AgentOutput(
            decision=decision,
            confidence=confidence,
            analysis=f"Rule workflow={workflow}, intent={intent_label}, query_type={query_type}, need_stage={need_stage}",
            user_facing=clarifying_question,
        )

    def update_state(self, state: SessionState, output: AgentOutput):
        if "need_stage" in output.decision:
            state.slots.need_stage = output.decision["need_stage"]
        if "intent_label" in output.decision:
            state.slots.extracted_features["intent_label"] = output.decision["intent_label"]
        if "workflow" in output.decision:
            state.slots.extracted_features["workflow"] = output.decision["workflow"]
        state.slots.extracted_features["intent_payload"] = output.decision
        if output.decision.get("user_goal"):
            state.slots.user_goal = output.decision.get("user_goal")
