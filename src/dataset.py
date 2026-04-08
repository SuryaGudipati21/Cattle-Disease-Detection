import os
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

# Class labels
CLASS_NAMES = ["healthy", "possibly_sick"]

def get_transforms(split="train"):
    """
    Returns transforms for train or val split.
    Train: augmentation + normalize
    Val:   only resize + normalize
    """
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )

    if split == "train":
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            normalize,
        ])
    else:
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            normalize,
        ])


class CattleDataset(Dataset):
    """
    Expects folder structure:
        data/train/healthy/
        data/train/possibly_sick/
        data/val/healthy/
        data/val/possibly_sick/
    """
    def __init__(self, root_dir, split="train", transform=None):
        self.root_dir = os.path.join(root_dir, split)
        self.transform = transform or get_transforms(split)
        self.samples = []
        self.labels = []

        for label_idx, class_name in enumerate(CLASS_NAMES):
            class_dir = os.path.join(self.root_dir, class_name)
            if not os.path.exists(class_dir):
                print(f"[WARNING] Missing folder: {class_dir}")
                continue
            for fname in os.listdir(class_dir):
                if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                    self.samples.append(os.path.join(class_dir, fname))
                    self.labels.append(label_idx)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img = Image.open(self.samples[idx]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, self.labels[idx]


def get_dataloaders(data_dir="data", batch_size=32):
    train_ds = CattleDataset(data_dir, split="train")
    val_ds   = CattleDataset(data_dir, split="val")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=2)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=2)

    print(f"Train samples: {len(train_ds)} | Val samples: {len(val_ds)}")
    return train_loader, val_loader


if __name__ == "__main__":
    train_loader, val_loader = get_dataloaders()
    imgs, labels = next(iter(train_loader))
    print("Batch shape:", imgs.shape)
    print("Labels:", labels)