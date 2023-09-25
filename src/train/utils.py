import os
import random
import shutil
from io import StringIO

import numpy as np
import pandas as pd
import torch
from sklearn.metrics.pairwise import paired_cosine_distances


class MetricTracker:
    def __init__(self) -> None:
        self.metrics = {}

    def log(self, name, value, epoch, step):
        if name in self.metrics:
            self.metrics[name].append({name: value, "epoch": epoch, "step": step})
        else:
            self.metrics[name] = [{name: value, "epoch": epoch, "step": step}]


def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)


def s3_save_csv(df, s3_client, target_bucket, target_key):
    csv_buffer = StringIO()
    df.to_csv(csv_buffer, index=False)
    s3_client.put_object(
        Bucket=target_bucket, Key=target_key, Body=csv_buffer.getvalue()
    )
    csv_buffer.close()


def s3_load_csv(s3_client, src_bucket, src_key):
    s3_response = s3_client.get_object(Bucket=src_bucket, Key=src_key)
    df = pd.read_csv(s3_response["Body"])
    return df


def s3_download_all(s3_client, src_bucket, src_key, tgt_local_directory):
    os.makedirs(tgt_local_directory, exist_ok=True)
    continuation_token = None
    while True:
        list_objects_kwargs = {"Bucket": src_bucket, "Prefix": src_key}

        if continuation_token:
            list_objects_kwargs["ContinuationToken"] = continuation_token

        response = s3_client.list_objects_v2(**list_objects_kwargs)
        if "Contents" in response:
            for obj in response["Contents"]:
                key = obj["Key"]
                local_file_path = os.path.join(tgt_local_directory, key[len(src_key) :])
                os.makedirs(os.path.dirname(local_file_path), exist_ok=True)
                s3_client.download_file(src_bucket, key, local_file_path)
        if "NextContinuationToken" in response:
            continuation_token = response["NextContinuationToken"]
        else:
            break


def delete_path(path):
    if os.path.isfile(path):
        os.remove(path)
    elif os.path.isdir(path):
        shutil.rmtree(path)


def get_cosine_similarity_scores_shuffled(
    anchor_embeddings, positive_embeddings, negative_embeddings
):
    pos_cosine_distances = 1 - paired_cosine_distances(
        anchor_embeddings, positive_embeddings
    )
    neg_cosine_distances = 1 - paired_cosine_distances(
        anchor_embeddings, negative_embeddings
    )
    return pd.DataFrame(
        {
            "score": np.concatenate((pos_cosine_distances, neg_cosine_distances)),
            "match": [1 for i in range(len(pos_cosine_distances))]
            + [0 for i in range(len(neg_cosine_distances))],
        }
    ).sample(frac=1)
