import os
import shutil

import requests
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
