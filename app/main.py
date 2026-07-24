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
import auth as user_auth

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

orchestrator = Orchestrator()

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
@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest, username: str = Depends(require_auth)):
    try:
        is_valid, error_msg = Guardrails.validate_message(request.message)
        if not is_valid:
            raise HTTPException(status_code=400, detail=error_msg)

        if not Guardrails.is_behavioral_science_related(request.message):
            reply = Guardrails.rejection_message()
            return ChatResponse(
                session_id=request.session_id or "default",
                reply=reply,
                debug={},
            )

        reply, debug_info = orchestrator.process_message(
            session_id=request.session_id or "default",
            user_message=request.message,
            workflow_override=request.workflow,
            uploaded_context_text=request.document_text,
        )

        reply = Guardrails.sanitize_response(reply)

        response = ChatResponse(
            session_id=request.session_id or "default",
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


@app.get("/sessions/{session_id}")
async def get_session(session_id: str, username: str = Depends(require_auth)):
    """Return session state (debug)."""
    from app.core.state_store import state_store

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
