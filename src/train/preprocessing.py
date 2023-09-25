import torch
import torchvision.transforms as transforms
from PIL import Image


def load_input_example(
    image_path,
    label,
    transform,
):
    image = Image.open(image_path).convert("RGB")
    image_input = transform(image).squeeze()
    label = torch.tensor(label)
    return {"image_input": image_input, "label": label}


def get_augmentations(
    level="none",
    resize_hw=(224, 224),
    norm_mean=(0.485, 0.456, 0.406),
    norm_std=(0.229, 0.224, 0.225),
):
    match level:
        case "none":
            return transforms.Compose(
                [
                    transforms.Resize(resize_hw),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=norm_mean, std=norm_std),
                ]
            )
        case "low":
            return transforms.Compose(
                [
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.ColorJitter(
                        brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1
                    ),
                    transforms.RandomRotation((0, 5)),
                    transforms.Resize(resize_hw),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=norm_mean, std=norm_std),
                ]
            )
        case "high":
            return transforms.Compose(
                [
                    transforms.RandomResizedCrop(
                        resize_hw, scale=(0.60, 1.0), ratio=(0.9, 1.1)
                    ),
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.ColorJitter(
                        brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2
                    ),
                    transforms.RandomRotation((0, 30)),
                    transforms.RandomPerspective(distortion_scale=0.2, p=0.5),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=norm_mean, std=norm_std),
                ]
            )
        case _:
            raise NotImplementedError(f"Not implemented {level} augmentation")
