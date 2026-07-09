from PIL import Image
from pathlib import Path
from transformers import AutoModel


import cv2
import mlflow.pytorch as mlflow_pytorch
import numpy as np
import torch
import transform


video_dir = Path("data/video")
input_file = video_dir / "IMG_0886.mov"
output_file = video_dir / "inference.mp4"
target_fps = 5
threshold = 0.9
resize_size = 512
alpha = 0.4

# load model
image_transform, _ = transform.get_transforms(resize_size)
device = "cuda" if torch.cuda.is_available() else "cpu"
backbone = AutoModel.from_pretrained("facebook/dinov3-vitl16-pretrain-lvd1689m").to(device)
head = mlflow_pytorch.load_model("models:/m-24a1de6331bf4762a2d26eacf236f93e").to(device)
backbone.eval()
head.eval()

# get video information
capture = cv2.VideoCapture(input_file)
if not capture.isOpened():
    raise RuntimeError("OpenCV could not open the video.")
original_fps = capture.get(cv2.CAP_PROP_FPS)
original_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
original_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frame = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
stride = max(1, round(original_fps / target_fps))

writer = cv2.VideoWriter(output_file, cv2.VideoWriter.fourcc(*"mp4v"), target_fps, (original_width, original_height))

index = 0
with torch.inference_mode():
    while True:
        ok, bgr_frame = capture.read()
        if not ok:
            break

        if index % stride != 0:
            index += 1
            continue
            
        # inference
        rgb_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        input = image_transform(Image.fromarray(rgb_frame)).unsqueeze(0).to(device)
        logits = head(backbone(input).last_hidden_state[:, 5:])
        prob = torch.sigmoid(logits)

        # resize
        prob = torch.nn.functional.interpolate(
            prob,
            size=(original_height, original_width),
            mode="bilinear",
            align_corners=False,
        )

        # overlay
        mask = (prob > threshold).squeeze().cpu().numpy()
        bgr_frame[mask] = ((1 - alpha) * bgr_frame[mask] + np.array([0, 255 * alpha, 0])).astype(np.int8)
        writer.write(bgr_frame)

        index += 1
        print(f"[{index}/{total_frame}]")