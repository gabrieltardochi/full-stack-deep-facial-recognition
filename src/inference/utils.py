import os
import shutil

import requests
import torch.nn as nn
from PIL import Image


def download_image(url, save_path):
    response = requests.get(url)
    with open(save_path, "wb") as file:
        file.write(response.content)


def read_image_from_disk(image_path):
    return Image.open(image_path).convert("RGB")


def delete_path(path):
    if os.path.isfile(path):
        os.remove(path)
    elif os.path.isdir(path):
        shutil.rmtree(path)


# Define a function to get the out_features of the last linear layer
def get_last_linear_out_features(model):
    last_linear = None

    for module in model.modules():
        if isinstance(module, nn.Linear):
            last_linear = module

    if last_linear is None:
        raise ValueError("No Linear (fully connected) layer found in the model.")

    return last_linear.out_features
