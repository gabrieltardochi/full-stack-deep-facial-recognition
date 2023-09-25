import torchvision.transforms as transforms
from PIL import Image


def prepare_input(image, resize_hw, normalization_mean, normalization_std):
    transform = transforms.Compose(
        [
            transforms.Resize(resize_hw),
            transforms.ToTensor(),
            transforms.Normalize(mean=normalization_mean, std=normalization_std),
        ]
    )
    image_input = transform(image).squeeze()
    return image_input
