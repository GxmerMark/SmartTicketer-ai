"""FastAPI application for SmartTicket AI."""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from smartticket.model import predict_ticket, train_models, DATA_PATH

app = FastAPI(
    title="SmartTicket AI",
    description="NLP-powered support ticket classification and triage API.",
    version="1.0.0",
)


class TicketRequest(BaseModel):
    message: str = Field(..., min_length=3, max_length=5000,
                         examples=["I was charged twice for my subscription."])


@app.get("/health")
def health():
    return {"status": "ok", "service": "SmartTicket AI"}


@app.post("/predict")
def predict(request: TicketRequest):
    try:
        return predict_ticket(request.message)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail="Prediction failed. Check server logs.") from exc


@app.post("/train")
def train():
    try:
        return train_models(DATA_PATH)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Training failed: {exc}") from exc
