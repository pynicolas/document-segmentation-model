# Copyright 2025 Pierre-Yves Nicolas

# This program is free software: you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the Free
# Software Foundation, either version 3 of the License, or (at your option)
# any later version.
# This program is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
# FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for
# more details.
# You should have received a copy of the GNU General Public License along with
# this program. If not, see <https://www.gnu.org/licenses/>.

import os
import cv2
import glob
import zipfile
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torchvision.transforms.functional as TF
import albumentations as A
from albumentations.pytorch import ToTensorV2
import segmentation_models_pytorch as smp
from sklearn.metrics import f1_score
from urllib.request import urlretrieve
import shutil

DATASET_ZIP_URL = 'https://github.com/pynicolas/document-segmentation-dataset/releases/download/v1.0/document-segmentation-dataset-v1.0.zip'

BUILD_DIR = "build"
MODEL_DIR = BUILD_DIR + "/model"
MODEL_FILE_PATH = MODEL_DIR + "/document-segmentation-model.pth"
TFLITE_MODEL_FILE_PATH = MODEL_DIR + "/document-segmentation-model.tflite"
DATASET_ZIP_PATH = BUILD_DIR + "/dataset.zip"
DATASET_PARENT_DIR = BUILD_DIR + "/dataset"
DATASET_DIR = DATASET_PARENT_DIR + "/document-segmentation-dataset"

if os.path.isdir(BUILD_DIR):
    shutil.rmtree(BUILD_DIR)
os.makedirs(BUILD_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# Dataset

print('Download and extract dataset...')
urlretrieve(DATASET_ZIP_URL, DATASET_ZIP_PATH)
with zipfile.ZipFile(DATASET_ZIP_PATH, 'r') as zip_ref:
    zip_ref.extractall(DATASET_PARENT_DIR)

class DocumentSegmentationDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_paths = sorted(glob.glob(os.path.join(image_dir, "*.jpg")))
        self.mask_paths = sorted(glob.glob(os.path.join(mask_dir, "*.png")))
        self.transform = transform

    def __getitem__(self, idx):
        image = cv2.imread(self.image_paths[idx])  # BGR
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(self.mask_paths[idx], cv2.IMREAD_GRAYSCALE)

        mask = (mask > 127).astype('float32')

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask'].unsqueeze(0)  # (1, H, W)
        else:
            image = TF.to_tensor(image)
            mask = TF.to_tensor(mask)

        return image, mask

    def __len__(self):
        return len(self.image_paths)


# Data loaders

print('Set up data loaders...')

shared_transform = A.Compose([
    A.Resize(256, 256),
    A.Normalize(),
    ToTensorV2(),
])

train_transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.2),
    shared_transform,
])

train_dataset = DocumentSegmentationDataset(
    image_dir=os.path.join(DATASET_DIR, "train/images"),
    mask_dir=os.path.join(DATASET_DIR, "train/masks"),
    transform=train_transform
)
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)

val_dataset = DocumentSegmentationDataset(
    image_dir=os.path.join(DATASET_DIR, "val/images"),
    mask_dir=os.path.join(DATASET_DIR, "val/masks"),
    transform=shared_transform
)
val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)


# Training

dice_loss = smp.losses.DiceLoss(mode='binary')
bce_loss = nn.BCEWithLogitsLoss()

def loss_fn(pred, target):
    return dice_loss(pred, target) + bce_loss(pred, target)

def evaluate_encoder(encoder_name, model_save_path, device=torch.device('cpu')):
    model = smp.Unet(
        encoder_name=encoder_name,
        encoder_weights="imagenet",
        in_channels=3,
        classes=1
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    best_dice = -1
    best_state = None
    nb_epochs = 20

    for epoch in range(nb_epochs):

        model.train()
        for images, masks in train_loader:
            images, masks = images.to(device), masks.to(device)
            preds = model(images)
            loss = loss_fn(preds, masks)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        model.eval()
        all_preds, all_targets = [], []
        with torch.no_grad():
            for images, masks in val_loader:
                images, masks = images.to(device), masks.to(device)
                preds = torch.sigmoid(model(images)) > 0.5
                all_preds.append(preds.cpu().numpy())
                all_targets.append(masks.cpu().numpy())

        # Calculate Dice score
        pred_flat = np.concatenate(all_preds).astype(np.uint8).ravel()
        target_flat = np.concatenate(all_targets).astype(np.uint8).ravel()
        dice = f1_score(target_flat, pred_flat)
        print(f"- Epoch {epoch + 1}/{nb_epochs}: train_loss={loss.item():.4f} dice={dice:.4f}")

        if dice > best_dice:
            best_dice = dice
            best_state = model.state_dict()

    # Save best model temporarily to measure size
    torch.save(best_state, model_save_path)
    print(f"Wrote {MODEL_FILE_PATH}")
    model_size_mb = os.path.getsize(model_save_path) / 1e6

    # Inference time test
    dummy_input = torch.randn(1, 3, 256, 256).to(device)
    model.eval()
    model.load_state_dict(best_state)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start_time = time.time()
    for _ in range(10):
        _ = model(dummy_input)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    inference_time_ms = (time.time() - start_time) * 100 / 10

    return {
        "encoder": encoder_name,
        "dice": round(best_dice, 4),
        "size_mb": round(model_size_mb, 2),
        "inference_ms": round(inference_time_ms, 2)
    }


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

encoder = "timm-tf_efficientnet_lite0"
# Other possible encoders:
# "timm-tf_efficientnet_lite1"
# "timm-efficientnet-b0"
# "mobilenet_v2"
# "mobileone_s0"

print(f"Training {encoder}...")
result = evaluate_encoder(encoder, model_save_path=MODEL_FILE_PATH, device=device)
print(result)


# Convert to TFLite

import ai_edge_torch
from ai_edge_torch.generative.quantize import quant_recipes

model = smp.Unet(
    encoder_name="timm-tf_efficientnet_lite0",
    encoder_weights=None,
    in_channels=3,
    classes=1,
)
model.load_state_dict(torch.load(MODEL_FILE_PATH, map_location="cpu"))
model.eval()

sample_inputs = next(iter(train_loader))[0]
sample_inputs = sample_inputs[:1]
sample_inputs = sample_inputs.to(torch.float32)
sample_args = (sample_inputs,)

quant_config = quant_recipes.full_int8_dynamic_recipe()
edge_model_quantized = ai_edge_torch.convert(
  model,
  sample_args,
  quant_config=quant_config
)
edge_model_quantized.export(TFLITE_MODEL_FILE_PATH)
print(f"Wrote {TFLITE_MODEL_FILE_PATH}")
