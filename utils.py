import torch
import torchvision
from data import CityscapesDataset
from torch.utils.data import DataLoader
import os
from PIL import Image
import numpy as np

def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)

def load_checkpoint(checkpoint, model):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])

def get_loaders(train_dir, train_maskdir, val_dir, val_maskdir, batch_size, train_transform, val_transform, num_workers=4, pin_memory=True):
    train_ds = CityscapesDataset(image_dir=train_dir, mask_dir=train_maskdir, transform=train_transform)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory)

    val_ds = CityscapesDataset(image_dir=val_dir, mask_dir=val_maskdir, transform=val_transform)
    val_loader = DataLoader(val_ds, batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory)

    return train_loader, val_loader

def check_accuracy(loader, model, device="cuda", num_classes=30):
    num_correct = 0
    num_pixels = 0
    model.eval()
    dice_scores = []

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)  # shape (N, H, W), dtype long

            preds = model(x)              # (N, C, H, W)
            preds = torch.argmax(preds, dim=1)  # (N, H, W)

            num_correct += (preds == y).sum().item()
            num_pixels  += torch.numel(preds)

            for cls in range(num_classes):
                pred_cls = (preds == cls).float()
                true_cls = (y == cls).float()

                intersection = (pred_cls * true_cls).sum()
                union = pred_cls.sum() + true_cls.sum()

                if union.item() > 0:  # avoid div by zero
                    dice = (2.0 * intersection) / union
                    dice_scores.append(dice.item())

    acc = 100.0 * num_correct / num_pixels
    mean_dice = sum(dice_scores) / len(dice_scores)

    print(f"Pixel Accuracy: {acc:.2f}%  ({num_correct}/{num_pixels})")
    print(f"Mean Dice Score: {mean_dice:.4f}")

    model.train()
    return acc, mean_dice

import random
def save_predictions_as_imgs(loader, model, folder="saved_images/", device="cuda", colors=None):
    """
    Save **one randomly chosen prediction** from validation set as RGB image.
    Also saves the corresponding ground truth.

    Args:
        loader: DataLoader
        model: trained segmentation model
        folder: output folder
        device: "cuda" or "cpu"
        colors: (num_classes,3) RGB array
    """
    os.makedirs(folder, exist_ok=True)
    model.eval()

    # Pick a random batch
    total_batches = len(loader)
    batch_idx = random.randint(0, total_batches-1)
    for idx, (x, y) in enumerate(loader):
        if idx != batch_idx:
            continue
        x = x.to(device)
        with torch.no_grad():
            preds = model(x)                  # (N, C, H, W)
            preds_idx = torch.argmax(preds, dim=1)  # (N, H, W)
            
            # Pick a random example from batch
            i = random.randint(0, preds_idx.shape[0]-1)
            pred_mask = preds_idx[i].cpu().numpy()  # (H, W)

            if colors is not None:
                rgb_mask = np.array(colors, dtype=np.uint8)[pred_mask]
                Image.fromarray(rgb_mask).save(os.path.join(folder, f"pred_random.png"))

                # ground truth
                y_i = y[i].cpu().numpy()
                if y_i.ndim == 3 and y_i.shape[0] == len(colors):
                    y_i = np.argmax(y_i, axis=0)
                true_rgb = np.array(colors, dtype=np.uint8)[y_i]
                Image.fromarray(true_rgb).save(os.path.join(folder, f"true_random.png"))
            else:
                Image.fromarray(pred_mask.astype(np.uint8)).save(os.path.join(folder, f"pred_random.png"))
        break  # exit after saving one example

    model.train()