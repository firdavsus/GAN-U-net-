import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms as T
from GAN import UNET
from utils import load_checkpoint

# ----------------------
# Device & Model
# ----------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = UNET(in_channels=3, out_channels=30).to(device)
load_checkpoint(torch.load("my_checkpoint.pth.tar"), model)
model.eval()

# ----------------------
# Colors for segmentation
# ----------------------
colors = np.array([
    (0, 0, 0), (111, 74, 0), (81, 0, 81), (128, 64, 128), (244, 35, 232), 
    (250, 170, 160), (230, 150, 140), (70, 70, 70), (102, 102, 156), (190, 153, 153), 
    (180, 165, 180), (150, 100, 100), (150, 120, 90), (153, 153, 153), (250, 170, 30), 
    (220, 220, 0), (107, 142, 35), (152, 251, 152), (70, 130, 180), (220, 20, 60), 
    (255, 0, 0), (0, 0, 142), (0, 0, 70), (0, 60, 100), (0, 0, 90), (0, 0, 110), 
    (0, 80, 100), (0, 0, 230), (119, 11, 32), (0, 0, 142)
], dtype=np.uint8)

# ----------------------
# Transform
# ----------------------
transform = T.Compose([
    T.ToTensor(),
    T.Resize((96, 256)),  # resize to model input
])

# ----------------------
# Load image
# ----------------------
img_path = "pic3.png"  # <-- specify your image path
img = Image.open(img_path).convert("RGB")
input_tensor = transform(img).unsqueeze(0).to(device)

# ----------------------
# Forward pass
# ----------------------
with torch.no_grad():
    preds = model(input_tensor)               # (1, C, H, W)
    preds_idx = torch.argmax(preds, dim=1)    # (1, H, W)
    pred_mask = preds_idx[0].cpu().numpy()    # (H, W)

# Map indices to colors
rgb_mask = colors[pred_mask]                 # (H, W, 3)

# ----------------------
# Display
# ----------------------
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.imshow(img)
plt.title("Original Image")
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(rgb_mask)
plt.title("GAN Segmentation")
plt.axis('off')

plt.show()
