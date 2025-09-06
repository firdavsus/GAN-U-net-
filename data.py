import os
from PIL import Image
from torch.utils.data import Dataset
import numpy as np
import torch

names = ['unlabeled', 'dynamic', 'ground', 'road', 'sidewalk', 'parking', 'rail track', 'building', 'wall',
             'fence', 'guard rail', 'bridge', 'tunnel', 'pole', 'traffic light', 'traffic sign', 'vegetation', 'terrain', 'sky', 'person', 'rider', 'car', 'truck',
             'bus', 'caravan', 'trailer', 'train', 'motorcycle', 'bicycle', 'license plate']

colors = np.array([(0, 0, 0), (111, 74, 0), (81, 0, 81), (128, 64, 128), (244, 35, 232), (250, 170, 160), (230, 150, 140), (70, 70, 70), 
            (102, 102, 156), (190, 153, 153), (180, 165, 180), (150, 100, 100), (150, 120, 90), (153, 153, 153), (250, 170, 30), (220, 220, 0), (107, 142, 35), 
            (152, 251, 152), (70, 130, 180), (220, 20, 60), (255, 0, 0), ( 0, 0, 142), ( 0, 0, 70), (0, 60, 100), (0, 0, 90), (0, 0, 110), (0, 80, 100), (0, 0, 230), 
            (119, 11, 32), (0, 0, 142)], dtype = np.int32)

def one_hot_mask_torch(y, colors):
    """
    Args:
        y: (H, W, 3) RGB mask, torch tensor or numpy
        colors: list/array of shape (num_classes, 3)
    Returns:
        mask: (num_classes, H, W) one-hot tensor
    """
    if isinstance(y, np.ndarray):
        y = torch.from_numpy(y)

    one_hot_map = []
    for color in colors:
        class_map = torch.all(y == torch.tensor(color, dtype=y.dtype), dim=-1)
        one_hot_map.append(class_map)

    mask = torch.stack(one_hot_map, dim=0).long()
    return mask

def snap_mask_to_colors(mask, colors):
    """
    Replace approximate RGB values in mask with the nearest color from colors.

    Args:
        mask: np.ndarray or torch.Tensor, shape (H, W, 3), dtype uint8 or int
        colors: array-like, shape (num_classes, 3), dtype uint8/int

    Returns:
        snapped_mask: np.ndarray, shape (H, W, 3), dtype uint8
    """
    # Convert to torch tensors
    if not torch.is_tensor(mask):
        mask_t = torch.from_numpy(mask.astype(np.int16))  # int16 to avoid overflow
    else:
        mask_t = mask.clone().to(torch.int16)

    colors_t = torch.from_numpy(np.array(colors, dtype=np.int16))  # (C, 3)

    H, W, _ = mask_t.shape
    mask_flat = mask_t.view(-1, 3)           # (H*W, 3)
    
    # Compute squared distance to each color: (H*W, C)
    dists = torch.cdist(mask_flat.float(), colors_t.float(), p=2)  # Euclidean
    idx_nearest = dists.argmin(dim=1)       # (H*W,)

    # Map each pixel to nearest color
    snapped_flat = colors_t[idx_nearest]     # (H*W, 3)
    snapped_mask = snapped_flat.view(H, W, 3).to(torch.uint8)

    return snapped_mask

class CityscapesDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = os.listdir(image_dir)

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, index):
        image_path = os.path.join(self.image_dir, self.images[index])
        mask_path  = os.path.join(self.mask_dir, self.images[index])

        # load image and mask from disk
        image = np.array(Image.open(image_path).convert("RGB"))
        mask  = np.array(Image.open(mask_path))

        snapped_mask = snap_mask_to_colors(mask, colors)

        # convert RGB mask → class indices 0..29
        mask = one_hot_mask_torch(snapped_mask, colors)     # (C, H, W)
        mask = torch.argmax(mask, dim=0).long()     # (H, W)

        # apply transforms (albumentations works with numpy arrays)
        if self.transform:
            aug = self.transform(image=image, mask=mask.numpy())
            image, mask = aug["image"], aug["mask"]

            # make sure mask is tensor (long for CE loss)
            if not torch.is_tensor(mask):
                mask = torch.from_numpy(mask).long()

        return image, mask