"""
FastAPI entrypoint
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from app.config import settings
from app.logging_config import logger
from app.core.types import ChatRequest, ChatResponse
from app.core.orchestrator import Orchestrator
from app.core.guardrails import Guardrails

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


@app.get("/")
async def root():
    return {
        "message": "NIH Stage Model AI Chatbot API",
        "version": settings.API_VERSION,
        "docs": "/docs",
    }


@app.get("/health")
async def health():
    return {"status": "healthy"}


@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    return Response(status_code=204)


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
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
async def get_session(session_id: str):
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
