from PIL import Image
from pathlib import Path
from typing import Callable
from torchmetrics import F1Score
from transformers import AutoModel
from torchinfo import summary
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import v2


import mlflow
import mlflow.pytorch as mlflow_pytorch
import os
import pandas as pd
import torch


# create custom dataset for ADE20K
class ADE20K_Nav(Dataset):
    def __init__(self,
                 data_dir: str,
                 split: str,
                 navigable_object: set[str],
                 image_transform: Callable[[Image.Image], torch.Tensor],
                 mask_transform: Callable[[Image.Image], torch.Tensor]) -> None:
        image_path = Path(data_dir) / "images" / split
        self.images = sorted(list(image_path.glob("*.jpg")), key=lambda p: p.name)

        label_path = Path(data_dir) / "annotations" / split
        self.labels = sorted(list(label_path.glob("*.png")), key=lambda p: p.name)

        object_info = pd.read_csv(Path(data_dir) / "objectInfo150.txt", sep="\t", engine="python")
        contain_navigable = object_info["Name"].apply(lambda names: bool(set(names.split(', ')) & navigable_object))
        self.navigable_indexes = list(object_info.index[contain_navigable] + 1)

        self.image_transform = image_transform
        self.mask_transform = mask_transform

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, index) -> tuple[torch.Tensor, torch.Tensor]:
        image, label = Image.open(self.images[index]), Image.open(self.labels[index])
        mask = torch.isin(self.mask_transform(label), torch.tensor(self.navigable_indexes), assume_unique=True).float()
        # image: (3, 256, 256)   label: (1, 256, 256)
        return (self.image_transform(image), mask)


class PixelClassification(nn.Module):
    def __init__(self, patch_size: int=16, hidden_size: int=1024) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.seq_layers = nn.Sequential(nn.Upsample(scale_factor=2, mode="bilinear"),
                                        nn.Conv2d(hidden_size, hidden_size // 4, kernel_size=1),
                                        nn.Upsample(scale_factor=2, mode="bilinear"),
                                        nn.Conv2d(hidden_size // 4, hidden_size // 16, kernel_size=1),
                                        nn.Upsample(scale_factor=2, mode="bilinear"),
                                        nn.Conv2d(hidden_size // 16, hidden_size // 64, kernel_size=1),
                                        nn.Upsample(scale_factor=patch_size // 8, mode="bilinear"),
                                        nn.Conv2d(hidden_size // 64, 1, kernel_size=1))
        
    def forward(self, dense_feature: torch.Tensor):
        _, token_length, hidden_size = dense_feature.shape
        side = int(token_length ** 0.5)
        dense_feature = torch.transpose(dense_feature, 1, 2).reshape((-1, hidden_size, side, side))
        return self.seq_layers(dense_feature)


def main():
    hparams = {
        "resize_size": 256,
        "batch_size": 64,
        "epoch": 5,
        "learning_rate": 0.01,
        "pos_weight": 9.0,
        "lr_start_factor": 1.0,
        "lr_end_factor": 0.03,
        "span": 1500
    }

    image_transform = v2.Compose([
        v2.ToImage(),
        v2.Resize((hparams["resize_size"], hparams["resize_size"]), antialias=True),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
        )
    ])

    mask_transform = v2.Compose([
        v2.ToImage(),
        v2.Resize((hparams["resize_size"], hparams["resize_size"]), antialias=False),
    ])

    navigable_object = {"floor", "flooring", "sidewalk",
                        "pavement", "earth", "ground",
                        "path", "step", "stair",
                        "rug", "carpet", "carpeting",
                        "stairway", "staircase", "stairs",
                        "steps"}

    # load dataset
    train_dataset = ADE20K_Nav("data/ADEChallengeData2016", "training", navigable_object, image_transform, mask_transform)
    validation_dataset = ADE20K_Nav("data/ADEChallengeData2016", "validation", navigable_object, image_transform, mask_transform)

    # create DataLoaders
    train_loader = DataLoader(train_dataset, hparams["batch_size"], shuffle=True, num_workers=os.cpu_count() or 1, pin_memory=True)
    validation_loader = DataLoader(validation_dataset, hparams["batch_size"], shuffle=True, num_workers=os.cpu_count() or 1, pin_memory=True)

    # load model and freeze model weights
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pretrained_model_name = "facebook/dinov3-vitl16-pretrain-lvd1689m"
    model = AutoModel.from_pretrained(pretrained_model_name).to(device)
    model.requires_grad_(False)
    summary(model, input_size=(hparams["batch_size"], 3, hparams["resize_size"], hparams["resize_size"]))

    # add model head
    head = PixelClassification(patch_size=model.config.patch_size, hidden_size=model.config.hidden_size).to(device)
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.SGD(head.parameters(), hparams["learning_rate"])
    scheduler = torch.optim.lr_scheduler.LinearLR(optimizer,
                                                  start_factor=hparams["lr_start_factor"],
                                                  end_factor=hparams["lr_end_factor"],
                                                  total_iters=hparams["span"])

    train_score = F1Score(task="binary").to(device)
    validation_score = F1Score(task="binary").to(device)

    mlflow.set_experiment("segmentation-conv-head")
    with mlflow.start_run(run_name="linear-lr", tags={"dataset": "ade20k"}):
        mlflow.log_params(hparams)
        # train the head with the loaded dataset
        for e in range(hparams["epoch"]):
            print(f"---------- epoch {e + 1} ----------")
            head.train()
            for i, (image, mask) in enumerate(train_loader):
                image = image.to(device, non_blocking=True)
                mask = mask.to(device, non_blocking=True)
                # skip class and register tokens
                logits = head(model(image).last_hidden_state[:, 5:])
                loss = loss_fn(logits, mask)

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                scheduler.step()

                mlflow.log_metric("train_loss", loss.item(), e * len(train_loader) + i)
                print(f"Train loss {loss.item():>7f}, [{i + 1}/{len(train_loader)}]")
                train_score.update(logits, mask.int())

            model.eval()
            head.eval()
            with torch.inference_mode():
                for i, (image, mask) in enumerate(validation_loader):
                    image = image.to(device, non_blocking=True)
                    mask = mask.to(device, non_blocking=True)
                    logits = head(model(image).last_hidden_state[:, 5:])
                    loss = loss_fn(logits, mask)
                    mlflow.log_metric("validation_loss", loss.item(), e * len(validation_loader) + i)

                    print(f"Validation loss {loss.item():>7f}, [{i + 1}/{len(validation_loader)}]")
                    validation_score.update(logits, mask.int())

            mlflow.log_metrics({"train_score": train_score.compute(), "validation_score": validation_score.compute()}, e)
            train_score.reset()
            validation_score.reset()

    mlflow_pytorch.log_model(head, name="dinov3-semseg-convhead", tags={"backbone": "dinov3", "task": "sem-seg", "dataset": "ade20k"})

if __name__ == "__main__":
    main()