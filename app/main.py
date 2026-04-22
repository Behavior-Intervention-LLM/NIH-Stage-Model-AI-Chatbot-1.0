"""
FastAPI entrypoint
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from app.config import settings
from app.logging_config import logger
from app.core.types import ChatRequest, ChatResponse
from app.core.orchestrator import Orchestrator
from app.core.guardrails import Guardrails

#  FastAPI 
app = FastAPI(
    title=settings.API_TITLE,
    version=settings.API_VERSION,
    debug=settings.DEBUG
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

orchestrator = Orchestrator()


# 
@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "NIH Stage Model AI Chatbot API",
        "version": settings.API_VERSION,
        "orchestration": "intent → stage → rag_agent → responder",
    }


@app.get("/health")
async def health():
    """Health check"""
    return {"status": "healthy"}


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Chat endpoint
    
    Args:
        request: Chat request
        
    Returns:
        ChatResponse: Chat response
    """
    try:
        # 1. Validate message (length / XSS)
        is_valid, error_msg = Guardrails.validate_message(request.message)
        if not is_valid:
            raise HTTPException(status_code=400, detail=error_msg)

        # 1b. Topic enforcement — only answer behavioral science questions
        if not Guardrails.is_behavioral_science_related(request.message):
            reply = Guardrails.rejection_message()
            return ChatResponse(
                session_id=request.session_id or "default",
                reply=reply,
                debug=None,
            )

        # 2. Process message
        reply, debug_info = orchestrator.process_message(
            session_id=request.session_id or "default",
            user_message=request.message,
            workflow_override=request.workflow,
            uploaded_context_text=request.document_text,
        )
        
        # 3. Sanitize response
        reply = Guardrails.sanitize_response(reply)
        
        # 4. Build response
        response = ChatResponse(
            session_id=request.session_id or "default",
            reply=reply,
            debug=debug_info
        )
        
        logger.info(f"Process message: session_id={response.session_id}, reply_length={len(reply)}")
        
        return response
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Process message: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@app.get("/sessions/{session_id}")
async def get_session(session_id: str):
    """Get session info (debug)"""
    from app.core.state_store import state_store
    
    state = state_store.get_state(session_id)
    if not state:
        raise HTTPException(status_code=404, detail="Session not found")
    
    return {
        "session_id": state.session_id,
        "message_count": len(state.messages),
        "slots": state.slots.dict(),
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
