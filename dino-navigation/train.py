from dataclasses import dataclass

from PIL import Image
from pathlib import Path
from typing import Any, Callable, Optional, cast
from torchmetrics import F1Score
from torchmetrics.classification import BinaryJaccardIndex
from transformers import AutoModel
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import v2


import argparse
import copy
import mlflow
import mlflow.pytorch as mlflow_pytorch
import os
import pandas as pd
import random
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
    def __init__(self, patch_size: int=16, hidden_size: int=1024, kernel_size: int=1) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.seq_layers = nn.Sequential(nn.Upsample(scale_factor=2, mode="bilinear"),
                                        nn.Conv2d(hidden_size, hidden_size // 4, kernel_size=kernel_size, padding="same"),
                                        nn.Upsample(scale_factor=2, mode="bilinear"),
                                        nn.Conv2d(hidden_size // 4, hidden_size // 16, kernel_size=kernel_size, padding="same"),
                                        nn.Upsample(scale_factor=2, mode="bilinear"),
                                        nn.Conv2d(hidden_size // 16, hidden_size // 64, kernel_size=kernel_size, padding="same"),
                                        nn.Upsample(scale_factor=patch_size // 8, mode="bilinear"),
                                        nn.Conv2d(hidden_size // 64, 1, kernel_size=kernel_size, padding="same"))
        
    def forward(self, dense_feature: torch.Tensor):
        _, token_length, hidden_size = dense_feature.shape
        side = int(token_length ** 0.5)
        dense_feature = torch.transpose(dense_feature, 1, 2).reshape((-1, hidden_size, side, side))
        return self.seq_layers(dense_feature)


class DiceWithLogitsLoss(nn.Module):
    def __init__(self, eps: float=1e-6):
        # epsilon for numerical stability
        super().__init__()
        self.eps = eps
    
    def forward(self, predict: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        predict = predict.sigmoid()
        label = label.float()
        dims = (1, 2, 3)
        true_positive = (predict * label).sum(dim=dims)
        dice = (2 * true_positive + self.eps) / (predict.sum(dim=dims) + label.sum(dim=dims) + self.eps)
        
        return 1 - dice.mean()


class EarlyStop:
    def __init__(self,
                 patient: int,
                 delta: float,
                 less: Callable[[float, float], bool],
                 best_metric: float):
        self.patient = patient
        self.delta = delta
        self.less = less
        self.best_metric = best_metric
        self.attempts = 0
    
    def stop(self, metric: float) -> bool:
        if self.less(metric, self.best_metric + self.delta):
            self.attempts += 1
        else:
            self.attempts = 0
            self.best_metric = metric
            
        return self.attempts >= self.patient


@dataclass
class State:
    rng_state: tuple[Any, ...]
    torch_rng_state: torch.Tensor
    cuda_rng_state: list[torch.Tensor]
    hparams: dict[str, Any]
    model_state: dict[str, Any]
    optimizer_state: dict[str, Any]
    round: int
    epoch: int
    metric: float
    attempts: int


class CheckpointManager:
    def __init__(self, checkpoint_dir: str, less: Callable[[float, float], bool]):
        directory = Path(checkpoint_dir)
        directory.mkdir(parents=True, exist_ok=True)
        self.best_path = directory / "best.pt"
        self.latest_path = directory / "latest.pt"
        self.less = less
        self.best_state: Optional[State] = None
        self.latest_state: Optional[State] = None

    def get_latest_state(self):
        if self.latest_state:
            return self.latest_state
        
        self.latest_state = cast(State, torch.load(self.latest_path, weights_only=False))
        return self.latest_state
    
    def get_best_state(self):
        if self.best_state:
            return self.best_state
        
        self.best_state = cast(State, torch.load(self.best_path, weights_only=False))
        return self.best_state
    
    def restore(self,
                model: nn.Module,
                optimizer: torch.optim.Optimizer,
                hparam_rng: random.Random,
                early_stop: EarlyStop):
        if self.latest_state is None:
            self.latest_state = cast(State, torch.load(self.latest_path, weights_only=False))

        model.load_state_dict(self.latest_state.model_state)
        optimizer.load_state_dict(self.latest_state.optimizer_state)
        hparam_rng.setstate(self.latest_state.rng_state)
        torch.set_rng_state(self.latest_state.torch_rng_state)
        torch.cuda.set_rng_state_all(self.latest_state.cuda_rng_state)

        if self.best_state is None:
            self.best_state = cast(State, torch.load(self.best_path, weights_only=False))
        
        early_stop.attempts = self.latest_state.attempts
        early_stop.best_metric = self.best_state.metric
        

    def update(self,
               model: nn.Module,
               optimizer: torch.optim.Optimizer,
               hparam_rng: random.Random,
               hparams: dict[str, Any],
               round: int,
               epoch: int,
               metric: float,
               attempts: int):
        self.latest_state = State(hparam_rng.getstate(),
                                  torch.get_rng_state(),
                                  torch.cuda.get_rng_state_all(),
                                  hparams, model.state_dict(),
                                  optimizer.state_dict(),
                                  round,
                                  epoch,
                                  metric,
                                  attempts)

        if self.best_state is None or not self.less(metric, self.best_state.metric):
            self.latest_state.attempts = 0
            self.best_state = copy.deepcopy(self.latest_state)
        
        torch.save(self.latest_state, self.latest_path)
        torch.save(self.best_state, self.best_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", default=None, type=str)
    args = parser.parse_args()

    comparator = lambda metric, best: metric < best
    cp_manager = CheckpointManager("data/checkpoints", comparator)
    current_state = None

    if args.run_id:
        current_state = cp_manager.get_latest_state()
        
    hparam_seed = 42
    train_seed = 654
    hparam_rng = random.Random(hparam_seed)
    torch.manual_seed(train_seed)

    batch_choices = {
        128: [32, 64, 128],
        256: [16, 32, 64],
        512: [8, 16, 32],
    }

    for r in range(current_state.round if current_state else 0, 30):
        if current_state is not None and r == current_state.round:
            hparams = current_state.hparams
        else:
            resize_size = hparam_rng.choice([128, 256, 512])
            hparams = {
                "resize_size": resize_size,
                "batch_size": hparam_rng.choice(batch_choices[resize_size]),
                "epoch": 20,
                "learning_rate": 10 ** hparam_rng.uniform(-5, -3),
                "pos_weight": 9.0,
                "patient": 3,
                "min_delta": 0.001,
                "predict_threshold": 0.9,
                "kernel_size": 3,
                "beta1": 0.9,
                "beta2": 0.999,
                "lambda": hparam_rng.uniform(0.0, 1.0),
                "hparam_seed": hparam_seed,
                "train_seed": train_seed,
            }

        image_transform = v2.Compose([
            v2.ToImage(),
            v2.Resize((hparams["resize_size"], hparams["resize_size"]), antialias=True),   # antialias help smoothen images
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            )
        ])

        mask_transform = v2.Compose([
            v2.ToImage(),
            v2.Resize((hparams["resize_size"], hparams["resize_size"]), interpolation=v2.InterpolationMode.NEAREST_EXACT, antialias=False),
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
        validation_loader = DataLoader(validation_dataset, hparams["batch_size"], shuffle=False, num_workers=os.cpu_count() or 1, pin_memory=True)

        # load model and freeze model weights
        device = "cuda" if torch.cuda.is_available() else "cpu"
        pretrained_model_name = "facebook/dinov3-vitl16-pretrain-lvd1689m"
        model = AutoModel.from_pretrained(pretrained_model_name).to(device)
        model.requires_grad_(False)
        model.eval()

        # add model head
        head = PixelClassification(patch_size=model.config.patch_size, hidden_size=model.config.hidden_size, kernel_size=hparams["kernel_size"]).to(device)
        # weighted bce loss is optimized to make each pixel's probability correct, with extra weight on positives
        bce_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([hparams["pos_weight"]], device=device))
        # dice loss is optimized to make the predicted navigable region overlap the true navigable region
        dice_fn = DiceWithLogitsLoss()
        optimizer = torch.optim.AdamW(head.parameters(), hparams["learning_rate"], betas=(hparams["beta1"], hparams["beta2"]), weight_decay=hparams["lambda"])
        
        run_id = None
        epoch_start = 0
        early_stop = EarlyStop(hparams["patient"], hparams["min_delta"], comparator, float("-inf"))
        if current_state is not None and r == current_state.round:
            run_id = args.run_id
            cp_manager.restore(head, optimizer, hparam_rng, early_stop)
            epoch_start = current_state.epoch + 1

        train_score = F1Score(task="binary", threshold=hparams["predict_threshold"]).to(device)
        validation_score = F1Score(task="binary", threshold=hparams["predict_threshold"]).to(device)
        train_positive_iou = BinaryJaccardIndex(threshold=hparams["predict_threshold"]).to(device)
        validation_positive_iou = BinaryJaccardIndex(threshold=hparams["predict_threshold"]).to(device)
        
        mlflow.set_experiment("segmentation-conv-head")
        run = f"sz={hparams['resize_size']}_b={hparams['batch_size']}_lr={round(hparams['learning_rate'], 5)}_rg={round(hparams['lambda'], 2)}"
        run_tags = {"dataset": "ade20k", "head": "3x3-conv", "schedule": "constant"}
        # need to set a checkpoint before mlflow.log_params() or it mlflow will complain "Changing param values is not allowed"
        # if the previous run is failed during the first epoch
        cp_manager.update(head, optimizer, hparam_rng, hparams, r, 0, float("-inf"), 0)

        with mlflow.start_run(run_name=run, run_id=run_id, tags=run_tags, log_system_metrics=True):
            mlflow.log_params(hparams)
            # train the head with the loaded dataset
            for e in range(epoch_start, hparams["epoch"]):
                print(f"---------- epoch {e + 1} ----------")
                head.train()
                for i, (image, mask) in enumerate(train_loader):
                    image = image.to(device, non_blocking=True)
                    mask = mask.to(device, non_blocking=True)
                    # skip class and register tokens
                    logits = head(model(image).last_hidden_state[:, 5:])
                    loss = bce_fn(logits, mask) + dice_fn(logits, mask)

                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    optimizer.step()

                    mlflow.log_metric("train_loss", loss.item(), e * len(train_loader) + i)
                    print(f"Train loss {loss.item():>7f}, [{i + 1}/{len(train_loader)}]")
                    train_score.update(logits, mask.int())
                    train_positive_iou.update(logits, mask.int())

                head.eval()
                with torch.inference_mode():
                    for i, (image, mask) in enumerate(validation_loader):
                        image = image.to(device, non_blocking=True)
                        mask = mask.to(device, non_blocking=True)
                        logits = head(model(image).last_hidden_state[:, 5:])
                        loss = bce_fn(logits, mask) + dice_fn(logits, mask)
                        mlflow.log_metric("validation_loss", loss.item(), e * len(validation_loader) + i)

                        print(f"Validation loss {loss.item():>7f}, [{i + 1}/{len(validation_loader)}]")
                        validation_score.update(logits, mask.int())
                        validation_positive_iou.update(logits, mask.int())

                stop_metric = validation_positive_iou.compute().item()
                mlflow.log_metrics({"train_score": train_score.compute(),
                                    "validation_score": validation_score.compute(),
                                    "train_iou": train_positive_iou.compute().item(),
                                    "validation_iou": stop_metric}, e)
                
                should_stop = early_stop.stop(stop_metric)
                cp_manager.update(head, optimizer, hparam_rng, hparams, r, e, stop_metric, early_stop.attempts)
                
                if should_stop:
                    break

                train_score.reset()
                validation_score.reset()
                train_positive_iou.reset()
                validation_positive_iou.reset()

            head.load_state_dict(cp_manager.get_best_state().model_state)
            mlflow_pytorch.log_model(head, name="dinov3-semseg-convhead", tags={"backbone": "dinov3", "task": "sem-seg", "dataset": "ade20k"})

if __name__ == "__main__":
    main()