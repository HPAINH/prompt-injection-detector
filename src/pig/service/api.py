from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from pydantic import BaseModel, Field

from pig import __version__
from pig.config import load_config
from pig.service.factory import Guard, make_guard
from pig.service.incidents import IncidentLog

CFG = load_config()
MAX_LEN = CFG["service"]["max_text_len"]


class CheckIn(BaseModel):
    text: str = Field(..., min_length=1, max_length=MAX_LEN)


class CheckOut(BaseModel):
    verdict: str
    score: float
    attack_type: str
    model: str
    latency_ms: float


class ChatOut(BaseModel):
    blocked: bool
    answer: str
    check: CheckOut


def llm_stub(text: str) -> str:
    return f"[LLM-заглушка] Ответ на вопрос: «{text[:200]}»"


def create_app(guard: Guard | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.guard = guard or make_guard(CFG)
        app.state.incidents = IncidentLog(CFG.path("logs") / "incidents.jsonl")
        yield

    app = FastAPI(title="PI-Guard", version=__version__, lifespan=lifespan)

    def run_check(request: Request, text: str) -> dict:
        res = request.app.state.guard.check(text)
        if res["verdict"] == "attack":
            request.app.state.incidents.write(text, res)
        return res

    @app.get("/health")
    def health(request: Request) -> dict:
        return {"status": "ok", "detector": request.app.state.guard.detector.name, "version": __version__}

    @app.post("/check", response_model=CheckOut)
    def check(body: CheckIn, request: Request) -> dict:
        return run_check(request, body.text)

    @app.post("/chat", response_model=ChatOut)
    def chat(body: CheckIn, request: Request) -> dict:
        res = run_check(request, body.text)
        blocked = res["verdict"] == "attack"
        answer = CFG["service"]["refusal"] if blocked else llm_stub(body.text)
        return {"blocked": blocked, "answer": answer, "check": res}

    return app


app = create_app()
