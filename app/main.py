"""
FastAPI entrypoint
"""
from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel
from app.config import settings
from app.logging_config import logger
from app.core.types import ChatRequest, ChatResponse, RatingRequest
from app.core.orchestrator import Orchestrator
from app.core.guardrails import Guardrails
from app import feedback
from app.feedback import adaptation as feedback_adaptation
from app.feedback import judge as feedback_judge
from app.feedback import rankings as feedback_rankings
from app.feedback import store as feedback_store
from app.tools import tool_registry
import auth as user_auth
import chat_history

app = FastAPI(
    title=settings.API_TITLE,
    version=settings.API_VERSION,
    debug=settings.DEBUG
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

orchestrator = Orchestrator(tool_registry=tool_registry)

# First-deploy bootstrap: creates accounts from SEED_USERS env var if set.
if not settings.AUTH_DISABLED:
    _seeded = user_auth.seed_users_from_env()
    if _seeded:
        logger.info(f"Seeded {len(_seeded)} user account(s) from SEED_USERS.")

_bearer = HTTPBearer(auto_error=False)


class LoginRequest(BaseModel):
    username: str
    password: str


class LoginResponse(BaseModel):
    token: str
    expires_in_seconds: int


def require_auth(
    credentials: HTTPAuthorizationCredentials | None = Depends(_bearer),
) -> str:
    """Resolve the authenticated username, or raise 401.

    Set AUTH_DISABLED=true to bypass (local development only).
    """
    if settings.AUTH_DISABLED:
        return "anonymous"
    if credentials is None:
        raise HTTPException(status_code=401, detail="Missing bearer token.")
    username = user_auth.verify_token(credentials.credentials)
    if username is None:
        raise HTTPException(status_code=401, detail="Invalid or expired token.")
    return username


@app.post("/login", response_model=LoginResponse)
async def login(request: LoginRequest):
    ok, msg = user_auth.verify_login(request.username, request.password)
    if not ok:
        raise HTTPException(status_code=401, detail=msg)
    token = user_auth.issue_token(request.username)
    return LoginResponse(token=token, expires_in_seconds=user_auth.TOKEN_TTL_SECONDS)


@app.post("/logout")
async def logout(credentials: HTTPAuthorizationCredentials | None = Depends(_bearer)):
    if credentials is not None:
        user_auth.revoke_token(credentials.credentials)
    return {"status": "logged out"}


@app.get("/")
async def root():
    return {
        "message": "NIH Stage Model AI Chatbot API",
        "version": settings.API_VERSION,
        "docs": "/docs",
        # "chat": "/chat"
    }


@app.get("/health")
async def health():
    return {"status": "healthy"}


@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    return Response(status_code=204)

#TODO: Logging 
def _record_history(username: str, session_id: str, message: str, reply: str) -> None:
    """Persist an exchange; history failures must never break the chat itself."""
    try:
        chat_history.record_exchange(username, session_id, message, reply)
    except Exception:
        logger.warning("Failed to record chat history", exc_info=True)


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest, username: str = Depends(require_auth)):
    try:
        session_id = request.session_id or "default"
        owner = chat_history.owner_of(session_id)
        if owner is not None and owner != username:
            raise HTTPException(status_code=403, detail="Session belongs to another user.")

        is_valid, error_msg = Guardrails.validate_message(request.message)
        if not is_valid:
            raise HTTPException(status_code=400, detail=error_msg)

        if not Guardrails.is_behavioral_science_related(request.message):
            reply = Guardrails.rejection_message()
            _record_history(username, session_id, request.message, reply)
            return ChatResponse(
                session_id=session_id,
                reply=reply,
                debug={},
            )

        reply, debug_info = orchestrator.process_message(
            session_id=session_id,
            user_message=request.message,
            workflow_override=request.workflow,
            uploaded_context_text=request.document_text,
            uploaded_context_name=request.document_name,
            username=username,
        )

        reply = Guardrails.sanitize_response(reply)
        _record_history(username, session_id, request.message, reply)

        response = ChatResponse(
            session_id=session_id,
            reply=reply,
            debug=debug_info,
            turn_uid=debug_info.get("turn_uid"),
        )

        logger.info(f"Chat: session={response.session_id}, reply_len={len(reply)}")

        return response

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Chat error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@app.get("/conversations")
async def list_conversations(username: str = Depends(require_auth)):
    """List the authenticated user's conversations, most recent first."""
    return {"conversations": chat_history.list_conversations(username)}


@app.get("/conversations/{conversation_id}/messages")
async def get_conversation_messages(
    conversation_id: str, username: str = Depends(require_auth)
):
    """Return the messages of a conversation the authenticated user owns."""
    messages = chat_history.get_messages(username, conversation_id)
    if messages is None:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return {"conversation_id": conversation_id, "messages": messages}


@app.delete("/conversations/{conversation_id}")
async def delete_conversation(
    conversation_id: str, username: str = Depends(require_auth)
):
    """Delete a conversation the authenticated user owns."""
    if not chat_history.delete_conversation(username, conversation_id):
        raise HTTPException(status_code=404, detail="Conversation not found")
    return {"status": "deleted"}


@app.post("/feedback/rating")
async def submit_rating(
    request: RatingRequest, username: str = Depends(require_auth)
):
    """Record a thumbs up/down (and optional comment) on one answer.

    `turn_uid` comes from the chat response that produced the answer.

    Send rating=null to withdraw a previous rating. Re-rating the same turn
    overwrites, so this is safe to call repeatedly from a toggle.

    The write replaces the whole rating: an omitted `comment` clears any
    comment already stored. Resend it alongside the rating to keep it.
    """
    try:
        return feedback.record_rating(
            turn_uid=request.turn_uid,
            username=username,
            rating=request.rating,
            comment=request.comment,
        )
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc))


@app.get("/feedback/rating/{turn_uid}")
async def read_rating(turn_uid: str, username: str = Depends(require_auth)):
    """Current rating for a turn the caller owns, or null."""
    owner = feedback_store.turn_owner(turn_uid)
    if owner is not None and owner != username.strip().lower():
        raise HTTPException(status_code=403, detail="That turn belongs to another user.")
    return {"turn_uid": turn_uid, "rating": feedback.get_rating(turn_uid)}


def require_admin(username: str = Depends(require_auth)) -> str:
    """Gate the analytics surface: it exposes per-user activity.

    Allow-list lives in ANALYTICS_ADMIN_USERS. See app.feedback.is_admin.
    """
    if not feedback.is_admin(username):
        raise HTTPException(
            status_code=403,
            detail="Analytics access requires membership in ANALYTICS_ADMIN_USERS.",
        )
    return username


@app.get("/analytics/overview")
async def analytics_overview(_: str = Depends(require_admin)):
    """Headline quality numbers and how much of the traffic is actually scored."""
    return feedback_rankings.overview()


@app.get("/analytics/rankings/features")
async def analytics_features(_: str = Depends(require_admin)):
    """Ranking of use: workflows, intents, query types, stages."""
    return feedback_rankings.feature_ranking()


@app.get("/analytics/rankings/users")
async def analytics_users(_: str = Depends(require_admin)):
    """Ranking of user usage: volume, breadth, and how well each user is served."""
    return {"users": feedback_rankings.user_ranking()}


@app.get("/analytics/rankings/responses")
async def analytics_responses(
    order: str = "worst", top_n: int = 20, _: str = Depends(require_admin)
):
    """Ranking of individual responses. order=worst is the fix-it queue."""
    if order not in {"worst", "best"}:
        raise HTTPException(status_code=400, detail="order must be 'worst' or 'best'")
    return {
        "order": order,
        "responses": feedback_rankings.response_ranking(top_n=max(1, min(top_n, 200)), order=order),
    }


@app.get("/analytics/rankings/sources")
async def analytics_sources(_: str = Depends(require_admin)):
    """Ranking of documents by learned contribution, with the weight in force."""
    return {"sources": feedback_rankings.source_ranking()}


@app.get("/analytics/needs")
async def analytics_needs(_: str = Depends(require_admin)):
    """What the system has inferred users are actually trying to accomplish."""
    return {"needs": feedback_rankings.inferred_user_needs()}


@app.get("/analytics/ratings")
async def analytics_ratings(
    limit: int = 200, comments_only: bool = False, _: str = Depends(require_admin)
):
    """Explicit user ratings, newest first. comments_only=true is the queue
    of answers someone took the trouble to write about."""
    return {
        "summary": feedback_rankings.rating_summary(),
        "ratings": feedback_store.rating_rows(
            limit=max(1, min(limit, 1000)), only_with_comments=comments_only
        ),
    }


@app.get("/analytics/gaps")
async def analytics_gaps(_: str = Depends(require_admin)):
    """Topics the corpus keeps answering badly — the ingestion to-do list."""
    return {"gaps": feedback_rankings.knowledge_gaps()}


@app.post("/analytics/recompute")
async def analytics_recompute(_: str = Depends(require_admin)):
    """Force a full relearn of source weights and knowledge gaps."""
    return feedback_adaptation.recompute_all()


@app.post("/analytics/judge-pending")
async def analytics_judge_pending(limit: int = 25, _: str = Depends(require_admin)):
    """Grade turns that have no judgement yet (e.g. after enabling the judge)."""
    return {"judged": feedback_judge.judge.judge_pending(limit=max(1, min(limit, 200)))}


@app.get("/sessions/{session_id}")
async def get_session(session_id: str, username: str = Depends(require_auth)):
    """Return session state (debug)."""
    from app.core.state_store import state_store

    # 404 (not 403) so the response never confirms another user's session exists.
    owner = chat_history.owner_of(session_id)
    if owner is not None and owner != username:
        raise HTTPException(status_code=404, detail="Session not found")

    state = state_store.get_state(session_id)
    if not state:
        raise HTTPException(status_code=404, detail="Session not found")

    return {
        "session_id": state.session_id,
        "message_count": len(state.messages),
        "slots": state.slots.model_dump(),
        "last_route": state.last_route,
        "artifact_count": len(state.artifacts)
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.DEBUG
    )
