from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Optional
from app.screener import evaluate_sanctions_prompt

app = FastAPI(
    title="Sanctions Screening Intelligence Miner",
    description="Deterministic AML/KYC sanctions entity resolution and false-positive disambiguation service.",
    version="1.0.0",
)


class ScreenRequest(BaseModel):
    prompt: Optional[str] = Field(None, description="Sanctions screening comparison prompt")
    query: Optional[str] = Field(None, description="Alternative query field")
    input: Optional[str] = Field(None, description="Alternative input field")


@app.get("/health")
def health():
    return {"status": "online", "intent": "SANCTIONS_SCREENING_MATCH_GENERAL"}


@app.post("/screen")
def screen_entity(payload: ScreenRequest):
    text = payload.prompt or payload.query or payload.input
    if not text or not text.strip():
        raise HTTPException(status_code=400, detail="Missing 'prompt' field in request body.")
    return evaluate_sanctions_prompt(text.strip())