from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from pydantic import BaseModel
import uvicorn

from src.pipeline.prediction_pipeline import PredictionPipeline


app = FastAPI(
    title="SmartSummarizer",
    description="An API for training and inference",
    version="1.0.0",
)

predictor = PredictionPipeline()


class SummaryRequest(BaseModel):
    text: str


class SummaryResponse(BaseModel):
    summary: str


@app.get("/", tags=["meta"])
async def homepage():
    # redirect root ("/") to the docs UI
    return RedirectResponse(url="/docs")


@app.post("/predict", response_model=SummaryResponse, tags=["inference"])
async def predict_route(request: SummaryRequest):
    summary = predictor.predict(request.text)
    return SummaryResponse(summary=summary)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080, reload=False)
