from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from pydantic import BaseModel
from usage import usage
from save_load import load_model

import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model
    try:
        model = load_model("checkpoints/best_checkpoint.pt")
        model.eval()
        logger.info("Model loaded successfully from checkpoints/best_model.pt")
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        raise
    yield
    model = None
    logger.info("Model unloaded during shutdown")

class Input(BaseModel):
    sentence: str

@app.get("/", response_class=HTMLResponse)
async def get_interface():
    try:
        with open("frontend.html", "r") as f:
            content = f.read()
        return HTMLResponse(content=content)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading frontend: {e}")

@app.post("/extract")
async def use_model(input: Input):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")
    try:
        print(f"Received input: {input.sentence}")
        output = str(usage(input.sentence, model)[0])[1:-1]
        print(f"Model output: {output}")
        
        return {"informationGraph": output}
    except Exception as e:
        print(f"Error during extraction: {e}")
        raise HTTPException(status_code=400, detail=f"Extraction failed: {str(e)}")