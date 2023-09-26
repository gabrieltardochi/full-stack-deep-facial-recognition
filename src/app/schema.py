from enum import Enum

from pydantic import BaseModel, Field


class ImageFormatEnum(str, Enum):
    jpg = "jpg"
    jpeg = "jpeg"
    png = "png"


class RecognizeInput(BaseModel):
    image_url: str = Field(
        ...,
        title="url of an image of a face",
        example="https://people.com/thmb/FCtpb7VRFLv1qirVB8wdyFQ-G5k=/1500x0/filters:no_upscale():max_bytes(150000):strip_icc():focal(979x425:981x427)/elon-musk-twitter-vote-for-step-down-121922-1-b04ccac7f93f4931be3567a2afca81e1.jpg",
    )
    image_format: ImageFormatEnum = Field(
        ..., title="image file format (png, jpg..)", example="jpg"
    )


class RecognizeResult(BaseModel):
    pred: str = Field(..., example="Unknown", title="recognized person, if any")
    proba: float = Field(
        ..., example=0.88, ge=0, le=1, title="recognition calibrated probability"
    )


class RecognizeResponse(BaseModel):
    error: bool = Field(..., example=False, title="whether there is an error")
    results: RecognizeResult = ...


class IndexInput(BaseModel):
    image_url: str = Field(
        ...,
        title="url of an image of a face",
        example="https://people.com/thmb/FCtpb7VRFLv1qirVB8wdyFQ-G5k=/1500x0/filters:no_upscale():max_bytes(150000):strip_icc():focal(979x425:981x427)/elon-musk-twitter-vote-for-step-down-121922-1-b04ccac7f93f4931be3567a2afca81e1.jpg",
    )
    image_format: ImageFormatEnum = Field(
        ..., title="image file format (png, jpg..)", example="jpg"
    )
    name: ImageFormatEnum = Field(
        ..., title="name of the person being indexed", example="gabriel tardochi"
    )


class IndexResult(BaseModel):
    msg: str = Field(..., example="Created", title="notify if it worked")


class IndexResponse(BaseModel):
    error: bool = Field(..., example=False, title="whether there is an error")
    results: IndexResult = ...


class ErrorResponse(BaseModel):
    error: bool = Field(..., example=True, title="whether there is error")
    message: str = Field(..., example="", title="error message")
    traceback: str = Field(None, example="", title="detailed traceback of the error")
