from pathlib import Path
import torch
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms
from src.config import VISION_RAW_DIR, VISION_IMG_SIZE, VISION_BATCH_SIZE, VISION_NUM_CLASSES

def _synthetic_vision():
    n_train, n_val = 256, 64
    h, w = VISION_IMG_SIZE
    x_train = torch.rand(n_train, 3, h, w)
    y_train = torch.randint(0, VISION_NUM_CLASSES, (n_train,))
    x_val = torch.rand(n_val, 3, h, w)
    y_val = torch.randint(0, VISION_NUM_CLASSES, (n_val,))
    return (
        DataLoader(TensorDataset(x_train, y_train), batch_size=VISION_BATCH_SIZE, shuffle=True),
        DataLoader(TensorDataset(x_val, y_val), batch_size=VISION_BATCH_SIZE, shuffle=False),
        VISION_NUM_CLASSES,
    )

def load_vision_datasets():
    train_dir = VISION_RAW_DIR / "train"
    val_dir = VISION_RAW_DIR / "val"
    if train_dir.exists() and val_dir.exists() and any(train_dir.iterdir()) and any(val_dir.iterdir()):
        tfm = transforms.Compose([
            transforms.Resize(VISION_IMG_SIZE),
            transforms.ToTensor(),
        ])
        train_ds = datasets.ImageFolder(str(train_dir), transform=tfm)
        val_ds = datasets.ImageFolder(str(val_dir), transform=tfm)
        train_loader = DataLoader(train_ds, batch_size=VISION_BATCH_SIZE, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_ds, batch_size=VISION_BATCH_SIZE, shuffle=False, num_workers=0)
        return train_loader, val_loader, len(train_ds.classes)
    return _synthetic_vision()
