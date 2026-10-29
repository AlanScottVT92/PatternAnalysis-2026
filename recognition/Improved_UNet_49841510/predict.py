from pathlib import Path

import torch
from torch.utils.data import DataLoader

from dataset import HipMRIDataset
from modules import UNet

NUM_CLASSES = 6
BATCH_SIZE = 6
TARGET_SIZE = (256, 144)
IGNORE_INDEX = -100

DATA_ROOT = Path(
    r"D:\academic\Semester 2 2026\COMP3710"
    r"\data_for_report(private)\keras_slices_data"
)

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

CHECKPOINT_PATH = (
    Path(__file__).resolve().parent
    / "checkpoints"
    / "best_unet_baseline_full_trained.pth"
)

test_dataset = HipMRIDataset(
    DATA_ROOT,
    "test",
    target_size=TARGET_SIZE
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

print("Test samples:", len(test_dataset))

model = UNet(
    in_channels=1,
    num_classes=NUM_CLASSES
).to(device)


checkpoint = torch.load(
    CHECKPOINT_PATH,
    map_location=device,
    weights_only=True
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.eval()

print("Loaded checkpoint epoch:", checkpoint["epoch"])
print("Checkpoint validation loss:", checkpoint["validation_loss"])

images, masks = next(iter(test_loader))

images = images.to(device)

with torch.inference_mode():
    logits = model(images)
    predictions = torch.argmax(logits, dim=1)

print("Input images:", images.shape)
print("Output logits:", logits.shape)
print("Predicted masks:", predictions.shape)
print("Ground truth masks:", masks.shape)
print("Predicted labels:", torch.unique(predictions).cpu())