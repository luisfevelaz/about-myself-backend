import logging

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware

from . import config
from .groq_client import generate_reply
from .models import ChatRequest, ChatResponse
from .rag import build_system_prompt
from .rate_limit import is_rate_limited

logger = logging.getLogger("uvicorn.error")

app = FastAPI(title="About Me Chat Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=config.ALLOWED_ORIGINS,
    allow_methods=["POST"],
    allow_headers=["Content-Type"],
)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/api/chat", response_model=ChatResponse)
def chat(payload: ChatRequest) -> ChatResponse:
    message = payload.message.strip()
    if not message:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="message must not be empty")

    if not config.GROQ_API_KEY:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="GROQ_API_KEY is not configured")

    identifier = payload.userReference or "anonymous"
    if is_rate_limited(identifier):
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Rate limit exceeded. Please wait a bit before sending more messages.",
        )

    # The frontend sends the full running history including the message just
    # typed; drop that duplicate and cap the rest to bound token usage.
    history = payload.history
    if history and history[-1].role == "user" and history[-1].content == payload.message:
        history = history[:-1]
    history = history[-config.MAX_HISTORY_MESSAGES :]
    history_payload = [{"role": item.role, "content": item.content} for item in history]

    system_prompt = build_system_prompt(message)

    try:
        reply = generate_reply(system_prompt, history_payload, message)
    except Exception:
        logger.exception("Groq request failed")
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Failed to generate a response",
        ) from None

    return ChatResponse(message=reply)
