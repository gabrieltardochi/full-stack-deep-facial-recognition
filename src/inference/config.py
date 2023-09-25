import os
from uuid import uuid4

import boto3
from metaflow import Step

from src.inference.model import init_model, load_calibrator, load_state_dict
from src.inference.utils import delete_path


class ProductionConfig:
    def __init__(
        self,
        model_name,
        model_init_kwargs,
        resize_hw,
        norm_mean,
        norm_std,
        s3_model_state_dict_bucket,
        s3_model_state_dict_key,
        s3_calibrator_bucket,
        s3_calibrator_key,
    ) -> None:
        self.model_name = model_name
        self.model_init_kwargs = model_init_kwargs
        self.resize_hw = resize_hw
        self.norm_mean = norm_mean
        self.norm_std = norm_std
        self.s3_model_state_dict_bucket = s3_model_state_dict_bucket
        self.s3_model_state_dict_key = s3_model_state_dict_key
        self.s3_calibrator_bucket = s3_calibrator_bucket
        self.s3_calibrator_key = s3_calibrator_key


def load(run):
    cfg = Step(f"FacialRecognitionTrainFlow/{run}/end").task.data.production_config

    rid = str(uuid4())
    base_path = f"/tmp/{rid}/"
    os.makedirs(base_path, exist_ok=True)

    s3_client = boto3.client(
        "s3",
        aws_access_key_id=os.environ["S3_ACCESS_KEY"],
        aws_secret_access_key=os.environ["S3_SECRET_KEY"],
        endpoint_url=os.environ["S3_URL"],
    )

    # encoder
    state_dict_path = os.path.join(base_path, "model_state_dict.pth")
    s3_client.download_file(
        cfg.s3_model_state_dict_bucket, cfg.s3_model_state_dict_key, state_dict_path
    )
    encoder = init_model(
        model_name=cfg.model_name, model_init_kwargs=cfg.model_init_kwargs
    )
    encoder.to("cpu")
    encoder = load_state_dict(model=encoder, state_dict_path=state_dict_path)

    # calibrator
    calibrator_path = os.path.join(base_path, "calibrator.joblib")
    s3_client.download_file(
        cfg.s3_calibrator_bucket, cfg.s3_calibrator_key, calibrator_path
    )
    calibrator = load_calibrator(calibrator_path=calibrator_path)

    delete_path(path=base_path)
    return {
        "encoder": encoder,
        "calibrator": calibrator,
        "resize_hw": cfg.resize_hw,
        "norm_mean": cfg.norm_mean,
        "norm_std": cfg.norm_std,
    }
