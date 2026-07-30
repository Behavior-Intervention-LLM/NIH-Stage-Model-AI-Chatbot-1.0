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
from app.core.types import ChatRequest, ChatResponse
from app.core.orchestrator import Orchestrator
from app.core.guardrails import Guardrails
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
        )

        reply = Guardrails.sanitize_response(reply)
        _record_history(username, session_id, request.message, reply)

        response = ChatResponse(
            session_id=session_id,
            reply=reply,
            debug=debug_info
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
