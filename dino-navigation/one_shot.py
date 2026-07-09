from PIL import Image
from pathlib import Path
from scipy.io import loadmat
from torch.utils.data import Dataset, DataLoader
from torchmetrics import F1Score
from torchmetrics.classification import BinaryJaccardIndex
from torchvision.transforms import v2
from transformers import AutoModel


import matplotlib.pyplot as plt
import mlflow.pytorch as mlflow_pytorch
import numpy as np
import os
import torch
import transform


class SUNRGBD(Dataset):
    def __init__(self,
                 data_dir: str,
                 image_transform: v2.Compose,
                 mask_transform: v2.Compose):
        mask_paths = Path(data_dir).rglob("seg.mat")

        self.dataset_paths = []
        for p in mask_paths:
            image = (p.parent / "image").glob("*.jpg")
            self.dataset_paths.append((next(image), p))

        self.navigable_object = {
            "floor", "dark_floor", "panalled_floor", "elevator_floor",
            "linoleum", "concrete_ground", "carpet", "carpets",
            "gray_carpet", "office_carpet", "another_carpet", "other_room_carpet",
            "carpet_area", "carpeted_flloor", "rug_or_carpet", "walk_way", "walkway",
            "corridor", "asile", "ramp", 'flloor',
            'fl;oor', 'floor mat', 'fdloor', 'white_tile',
            'mat', 'throw_rug', 'doormat', 'section_of_foor',
            'floor_mats', 'ground', 'fkoor', 'sidewalk',
            'floor_access_panel', 'yellow_rug', 'yoga mat', 'foot_path',
            'class_room_stage', 'black_tile', 'bathroom_mat', 'rug',
            'tilels', 'kitchen_mat', 'welcome_mat', 'footpath',
            'floor_mat', 'stage', 'colourful_mat',
        }
        self.image_transform = image_transform
        self.mask_transform = mask_transform

    def __len__(self) -> int:
        return len(self.dataset_paths)
    
    def __getitem__(self, index) -> tuple[torch.Tensor, torch.Tensor]:
        image_path, mask_path = self.dataset_paths[index]
        seg_data = loadmat(mask_path)
        names = [n.item() for n in seg_data["names"].reshape(-1)]
        navigable_index = [i + 1 for i in range(len(names)) if names[i] in self.navigable_object]
        image = self.image_transform(Image.open(image_path))
        mask = torch.tensor(np.isin(seg_data["seglabel"], navigable_index, assume_unique=True), dtype=torch.int)

        return (image, self.mask_transform(mask))

# create dataset and data loader
RESIZE_SIZE = 512
image_transform, mask_transform = transform.get_transforms(RESIZE_SIZE)
dataset = SUNRGBD("data/SUNRGBD", image_transform, mask_transform)
loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=os.cpu_count() or 1, pin_memory=True)

# load models
device = "cuda" if torch.cuda.is_available() else "cpu"
backbone = AutoModel.from_pretrained("facebook/dinov3-vitl16-pretrain-lvd1689m").to(device)
head = mlflow_pytorch.load_model("models:/m-24a1de6331bf4762a2d26eacf236f93e").to(device)

# create metrics
f1_score = F1Score(task="binary", threshold=0.9).to(device)
positive_iou = BinaryJaccardIndex(threshold=0.9).to(device)

# evaluate on the data
backbone.eval()
head.eval()
with torch.inference_mode():
    for i, (images, masks) in enumerate(loader):
        images = images.to(device)
        masks = masks.to(device)
        logits = head(backbone(images).last_hidden_state[:, 5:])
        f1_score.update(logits, masks)
        positive_iou.update(logits, masks)

        for image, mask in zip(images, masks):
            fig, axes = plt.subplots(1, 2, figsize=(12, 6))
            axes[0].imshow(image.squeeze().permute(1, 2, 0).cpu())
            axes[0].set_title("Original")
            axes[0].axis("off")

            axes[1].imshow(mask.cpu().squeeze())
            axes[1].set_title("Modified")
            axes[1].axis("off")

            plt.tight_layout()
            plt.show()
            
        print(f"[{i}/{len(loader)}]")

print(f"F1 score: {f1_score.compute()}")
print(f"IoU: {positive_iou.compute()}")
