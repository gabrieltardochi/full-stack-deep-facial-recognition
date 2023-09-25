import torchvision.transforms as transforms


def prepare_input(image, resize_hw, norm_mean, norm_std):
    transform = transforms.Compose(
        [
            transforms.Resize(resize_hw),
            transforms.ToTensor(),
            transforms.Normalize(mean=norm_mean, std=norm_std),
        ]
    )
    image_input = transform(image).unsqueeze(0)
    return image_input
