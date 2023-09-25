import joblib
import timm
import torch
import torch.nn as nn
from torch import Tensor


class FacialRecognitionNet(nn.Module):
    def __init__(self, model_name) -> None:
        super().__init__()
        self.timm_image_model_name = model_name
        self.image_model = timm.create_model(
            self.timm_image_model_name, pretrained=True, num_classes=0
        )

    def forward(self, image_input: Tensor) -> Tensor:
        image_features = self.image_model(image_input)
        return image_features.squeeze()


def load_calibrator(calibrator_path):
    return joblib.load(calibrator_path)


def init_model(model_name, model_init_kwargs):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model = FacialRecognitionNet(model_name, **model_init_kwargs)
    model.to(device)
    model.eval()
    return model


def load_state_dict(model, state_dict_path):
    model.load_state_dict(torch.load(state_dict_path))
    model.eval()
    return model


def save_state_dict(model, state_dict_path):
    model.eval()
    torch.save(model.state_dict(), state_dict_path)
