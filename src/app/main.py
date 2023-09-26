import os
from uuid import uuid4

import torch
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.logger import logger
from fastapi.middleware.cors import CORSMiddleware

from src.app.const import MF_RUN_ID
from src.app.schema import ErrorResponse, SearchInput, SearchResponse
from src.inference.config import load
from src.inference.preprocessing import prepare_input
from src.inference.utils import download_image, read_image_from_disk

load_dotenv(dotenv_path=".env")

# Initialize API Server
app = FastAPI(
    title="Facial Recognition API",
    description="AI-Powered Facial Recognition.",
    version="0.0.1",
    terms_of_service=None,
    contact=None,
    license_info=None,
)

# Allow CORS for local debugging
app.add_middleware(CORSMiddleware, allow_origins=["*"])


@app.on_event("startup")
async def startup_event():
    """
    Initialize FastAPI and add variables
    """

    logger.info("Starting API!")
    logger.info("Loading models..")
    app.package = load(run=MF_RUN_ID)


@app.post(
    "/api/v1/recognize",
    response_model=SearchResponse,
    responses={422: {"model": ErrorResponse}, 500: {"model": ErrorResponse}},
)
def recognize(request: Request, body: SearchInput):
    """
    Perform facial recognition given input data
    """

    logger.info("API predict called")
    logger.info(f"input: {body}")
    rid = str(uuid4())

    os.makedirs(f"/tmp/{rid}", exist_ok=True)

    image_path = f"/tmp/{rid}/image.{body.image_format}"

    # prepare data
    download_image(url=body.image_url, save_path=image_path)
    image = read_image_from_disk(image_path=image_path)

    # encode
    with torch.no_grad():
        # convert input from list to Tensor
        image_input = prepare_input(
            image=image,
            resize_hw=app.package["resize_hw"],
            norm_mean=app.package["norm_mean"],
            norm_std=app.package["norm_std"],
        )

        # encode
        embeddings = app.package["encoder"](image_input)

    # search people index
    # TODO

    # calibrate best matching cosine similarity
    # y_calib = app.package['calibrator'].transform([cos_sim])[0]

    # prepare json for returning
    results = {"proba": 0.7, "pred": "bla"}

    logger.info(f"results: {results}")

    return {"error": False, "results": results}


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8080,
        reload=True,
        debug=True,
    )
