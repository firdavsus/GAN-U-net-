import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import torch 
import torch.nn.functional as F
import matplotlib.pyplot as plt

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

def visualize_class(mask_one_hot, class_idx):
    """
    Visualize a single class from a one-hot mask in black & white.
    
    Args:
        mask_one_hot: (num_classes, H, W) tensor
        class_idx: integer, which class to visualize
        
    Returns:
        None (shows the image using matplotlib)
    """
    # Get the layer for the class
    class_layer = mask_one_hot[class_idx]  # (H, W)
    
    # Convert to 0-255 for black/white
    img = class_layer * 255
    img_np = img.cpu().numpy() if isinstance(img, torch.Tensor) else img

    plt.imshow(img_np, cmap='gray')
    plt.title(f"Class {class_idx}")
    plt.axis('off')
    plt.show()

colors = np.array([(0, 0, 0), (111, 74, 0), (81, 0, 81), (128, 64, 128), (244, 35, 232), (250, 170, 160), (230, 150, 140), (70, 70, 70), 
            (102, 102, 156), (190, 153, 153), (180, 165, 180), (150, 100, 100), (150, 120, 90), (153, 153, 153), (250, 170, 30), (220, 220, 0), (107, 142, 35), 
            (152, 251, 152), (70, 130, 180), (220, 20, 60), (255, 0, 0), ( 0, 0, 142), ( 0, 0, 70), (0, 60, 100), (0, 0, 90), (0, 0, 110), (0, 80, 100), (0, 0, 230), 
            (119, 11, 32), (0, 0, 142)], dtype = np.int32)


# changing colors (normalizing)
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



mask = np.array(Image.open("archive/train/label/train10.png"))
snapped_mask = snap_mask_to_colors(mask, colors)

def draw_mask_by_index(mask_idx, colors, title=None):
    if torch.is_tensor(mask_idx):
        mask_idx = mask_idx.cpu().numpy()
    colors_arr = np.array(colors, dtype=np.uint8)
    color_img = colors_arr[mask_idx]  # fancy indexing (H,W,3)
    plt.figure(figsize=(8,4))
    plt.imshow(color_img)
    if title:
        plt.title(title)
    plt.axis('off')
    plt.show()
    return color_img

mask = one_hot_mask_torch(snapped_mask, colors)
mask = torch.argmax(mask, dim=0).long()
draw_mask_by_index(mask, colors)