import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
from dataset import HipMRIDataset
from modules import UNet
import numpy as np
import nibabel as nib
import random

def calculate_class_metrics(prediction, target, class_id):
    pred_mask = (prediction == class_id)
    true_mask = (target == class_id)

    tp = (pred_mask & true_mask).sum().item()
    fp = (pred_mask & ~true_mask).sum().item()
    fn = (~pred_mask & true_mask).sum().item()

    return tp, fp, fn
# 1. Configuration
NUM_CLASSES = 6
BATCH_SIZE = 6
LEARNING_RATE = 1e-4
DATA_ROOT = Path(
        r"D:\academic\Semester 2 2026\COMP3710"
        r"\data_for_report(private)\keras_slices_data"
    )
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)
RANDOM_SEED = 42

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)

if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

CHECKPOINT_DIR = Path(__file__).resolve().parent / "checkpoints"
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

BEST_MODEL_PATH = CHECKPOINT_DIR / "best_unet_baseline.pth"

# 2. Load dataset
train_dataset = HipMRIDataset(DATA_ROOT, "train")

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0
)
validation_dataset = HipMRIDataset(
    DATA_ROOT,
    "validate",
    target_size=(256, 128)
)

validation_loader = DataLoader(
    validation_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)




# 3. Create model, loss function and optimizer


model = UNet(
    in_channels=1,
    num_classes=NUM_CLASSES
).to(device)

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=1e-4
)


NUM_EPOCHS = 3
MAX_BATCHES = 100
if device.type == "cuda":
    torch.cuda.reset_peak_memory_stats(device)

val_images, val_masks = next(iter(validation_loader))

print("===== Validation Batch =====")
print("Images:", val_images.shape)
print("Masks:", val_masks.shape)
print("Unique labels:", torch.unique(val_masks))


best_validation_loss = float("inf")
best_epoch = 0
for epoch in range(NUM_EPOCHS):

    model.train()
    total_loss = 0.0
    batch_count = 0

    for batch_idx, (images, masks) in enumerate(train_loader):

        if batch_idx >= MAX_BATCHES:
            break

        # 1. Check mask labels
        assert images.shape[1:] == (1, 256, 128)
        assert masks.shape[1:] == (256, 128)


        assert masks.min().item() >= 0
        assert masks.max().item() < NUM_CLASSES

        # 2. Move data to GPU
        images = images.to(device)
        masks = masks.to(device)

        # 3. Clear gradients
        optimizer.zero_grad()

        # 4. Forward Pass
        logits = model(images)

        # 5. Calculate Loss
        loss = criterion(logits, masks)

        # 6. Backward Pass
        loss.backward()

        # 7. Update parameters
        optimizer.step()

        # 8. Record Batch Loss
        total_loss += loss.item()
        batch_count += 1

    average_loss = total_loss / batch_count

    print(
        f"Epoch {epoch + 1}: "
        f"Average Loss = {average_loss:.4f}"
    )

    
    model.eval()
    validation_loss = 0.0
    validation_count = 0
    total_tp = [0] * NUM_CLASSES
    total_fp = [0] * NUM_CLASSES
    total_fn = [0] * NUM_CLASSES
    with torch.no_grad():

        for images, masks in validation_loader:
            
            # Check mask label range
            assert masks.min().item() >= 0
            assert masks.max().item() < NUM_CLASSES

            # Move tensors to GPU
            images = images.to(device)
            masks = masks.to(device)

            # Forward pass only
            logits = model(images)

            # Calculate validation loss
            loss = criterion(logits, masks)

            prediction = torch.argmax(logits, dim=1)

            for c in range(NUM_CLASSES):
                tp, fp, fn = calculate_class_metrics(prediction, masks, c)

                total_tp[c] += tp
                total_fp[c] += fp
                total_fn[c] += fn

            validation_loss += loss.item()
            validation_count += 1

    for c in range(NUM_CLASSES):
        tp = total_tp[c]
        fp = total_fp[c]
        fn = total_fn[c]
        predicted_pixels = tp + fp
        ground_truth_pixels = tp + fn

        print(
            f"Class {c}: "
            f"Predicted Pixels = {predicted_pixels}, "
            f"Ground Truth Pixels = {ground_truth_pixels}"
        )

        denominator_dice = 2 * tp + fp + fn
        denominator_iou = tp + fp + fn

        dice = (
            2 * tp / denominator_dice
            if denominator_dice > 0 else float("nan")
        )

        iou = (
            tp / denominator_iou
            if denominator_iou > 0 else float("nan")
        )

        print(f"Class {c}: Dice = {dice:.4f}, IoU = {iou:.4f}")

    average_validation_loss = validation_loss / validation_count


    print(
        f"Epoch {epoch + 1}: "
        f"Validation Loss = {average_validation_loss:.4f}"
    )

if average_validation_loss < best_validation_loss:

    best_validation_loss = average_validation_loss
    best_epoch = epoch + 1

    torch.save(
        {
            "epoch": best_epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "validation_loss": best_validation_loss,
            "num_classes": NUM_CLASSES,
            "random_seed": RANDOM_SEED,
        },
        BEST_MODEL_PATH
    )

    print(
        f"Best model saved! "
        f"Epoch = {best_epoch}, "
        f"Validation Loss = {best_validation_loss:.4f}"
    )

if device.type == "cuda":
    torch.cuda.synchronize(device)

    peak_memory = (
        torch.cuda.max_memory_allocated(device)
        / (1024 ** 2)
    )

    print(f"Peak GPU Memory: {peak_memory:.2f} MiB")

# Use your existing DATA_ROOT
validation_dataset = HipMRIDataset(DATA_ROOT, "validate")
test_dataset = HipMRIDataset(DATA_ROOT, "test")

for name, dataset in [
    ("Validation", validation_dataset),
    ("Test", test_dataset)
]:
    labels = set()

    for _, mask_path in dataset.pairs:
        mask = np.asanyarray(nib.load(str(mask_path)).dataobj)
        labels.update(np.unique(mask).tolist())

    print(f"{name} labels:", sorted(labels))


print("Best Epoch:", best_epoch)
print("Best Validation Loss:", best_validation_loss)
print("Checkpoint exists:", BEST_MODEL_PATH.is_file())