from torchvision.transforms import v2

import torch

def get_transforms(resize_size: int):
    image_transform = v2.Compose([
        v2.ToImage(),
        v2.Resize((resize_size, resize_size), antialias=True),   # antialias help smoothen images
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
        )
    ])

    mask_transform = v2.Compose([
        v2.ToImage(),
        v2.Resize((resize_size, resize_size), interpolation=v2.InterpolationMode.NEAREST_EXACT, antialias=False),
    ])

    return (image_transform, mask_transform)