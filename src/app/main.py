import os
from uuid import uuid4

import torch
import uvicorn
from elasticsearch import Elasticsearch
from fastapi import FastAPI, Request
from fastapi.logger import logger
from fastapi.middleware.cors import CORSMiddleware

from src.app.const import ES_INDEX, ES_URL, MF_RUN_ID
from src.app.schema import (
    ErrorResponse,
    IndexInput,
    IndexResponse,
    RecognizeInput,
    RecognizeResponse,
)
from src.inference.config import load
from src.inference.preprocessing import prepare_input
from src.inference.utils import download_image, read_image_from_disk

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
    app.package["elasticsearch"] = Elasticsearch(ES_URL)
    app.package["es_index_name"] = ES_INDEX


@app.post(
    "/api/v1/index",
    response_model=IndexResponse,
    responses={422: {"model": ErrorResponse}, 500: {"model": ErrorResponse}},
)
def index(request: Request, body: IndexInput):
    """
    Index a person's face so we can recognize later
    """

    logger.info("API index called")
    logger.info(f"input: {body}")
    rid = str(uuid4())

    os.makedirs(f"/tmp/{rid}", exist_ok=True)

    image_path = f"/tmp/{rid}/image.{body.image_format}"

    # prepare data
    download_image(url=body.image_url, save_path=image_path)
    image = read_image_from_disk(image_path=image_path)

    with torch.no_grad():
        # convert input to Tensor
        image_input = prepare_input(
            image=image,
            resize_hw=app.package["resize_hw"],
            norm_mean=app.package["norm_mean"],
            norm_std=app.package["norm_std"],
        )

        # encode
        embeddings = app.package["encoder"](image_input).cpu().numpy().tolist()

    doc = {"embeddings": embeddings, "name": str(body.name).lower().strip()}
    resp = app.package["elasticsearch"].index(
        index=app.package["es_index_name"], document=doc
    )
    # prepare json for returning
    results = {"msg": resp["result"]}

    logger.info(f"results: {results}")

    return {"error": False, "results": results}


@app.post(
    "/api/v1/recognize",
    response_model=RecognizeResponse,
    responses={422: {"model": ErrorResponse}, 500: {"model": ErrorResponse}},
)
def recognize(request: Request, body: RecognizeInput):
    """
    Perform facial recognition
    """

    logger.info("API recognize called")
    logger.info(f"input: {body}")
    rid = str(uuid4())

    os.makedirs(f"/tmp/{rid}", exist_ok=True)

    image_path = f"/tmp/{rid}/image.{body.image_format}"

    # prepare data
    download_image(url=body.image_url, save_path=image_path)
    image = read_image_from_disk(image_path=image_path)

    with torch.no_grad():
        # convert input to Tensor
        image_input = prepare_input(
            image=image,
            resize_hw=app.package["resize_hw"],
            norm_mean=app.package["norm_mean"],
            norm_std=app.package["norm_std"],
        )

        # encode
        embeddings = app.package["encoder"](image_input).cpu().numpy().tolist()

    # search people index
    resp = app.package["elasticsearch"].knn_search(
        index=app.package["es_index_name"],
        knn={
            "field": "embeddings",
            "query_vector": embeddings,
            "k": 1,
            "num_candidates": 100,
        },
        fields=["name"],
    )

    # calibrate best matching cosine similarity
    y_calib = app.package["calibrator"].transform([resp["hits"]["max_score"]])[0]
    name = resp["hits"]["hits"][0]["_source"]["name"]

    # prepare json for returning
    results = {"proba": y_calib, "pred": name}

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
