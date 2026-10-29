from pathlib import Path
import numpy as np
import nibabel as nib
import torch
from torch.utils.data import Dataset

class HipMRIDataset(Dataset):

    def __init__(self, root, split):
        self.root = Path(root)
        self.split = split

        # Select MRI and segmentation directories
        self.image_dir = self.root / f"keras_slices_{split}"
        self.mask_dir = self.root / f"keras_slices_seg_{split}"
        self.image_files = sorted(
            self.image_dir.glob("*.nii.gz")
        )

        self.pairs = []

        for image_path in self.image_files:

            mask_name = image_path.name.replace(
                "case_", "seg_", 1
            )

            mask_path = self.mask_dir / mask_name

            if not mask_path.is_file():
                raise FileNotFoundError(
                    f"Missing mask for {image_path.name}"
                )

            self.pairs.append((image_path, mask_path))

        
    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        image_path, mask_path = self.pairs[index]

        image = nib.load(str(image_path)).get_fdata()
        mask = nib.load(str(mask_path)).get_fdata()
        image = image.astype(np.float32)

        mean = image.mean()
        std = image.std()

        image = (image - mean) / (std + 1e-8)
        image = torch.from_numpy(image).unsqueeze(0)

        mask = torch.from_numpy(
            mask.astype(np.int64)
        )

        return image, mask

if __name__ == "__main__":

    DATA_ROOT = Path(
        r"D:\academic\Semester 2 2026\COMP3710"
        r"\data_for_report(private)\keras_slices_data"
    )

    train_dataset = HipMRIDataset(DATA_ROOT, "train")
    validation_dataset = HipMRIDataset(DATA_ROOT, "validate")
    test_dataset = HipMRIDataset(DATA_ROOT, "test")

    print("Training samples:", len(train_dataset))
    print("Validation samples:", len(validation_dataset))
    print("Test samples:", len(test_dataset))

    print("First pair:", train_dataset.pairs[0])
    image, mask = train_dataset[0]

    print("===== First Sample =====")
    print("Image shape:", image.shape)
    print("Image dtype:", image.dtype)

    print("Mask shape:", mask.shape)
    print("Mask dtype:", mask.dtype)

    print("Image mean:", image.mean().item())
    print("Image std:", image.std().item())

    print("Mask labels:", torch.unique(mask))