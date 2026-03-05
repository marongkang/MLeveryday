import torch
import torch.nn as nn
from DDPM import DDPM
from DDIM import DDIM
from DataLoader import get_dataloader
from DenoiseNetwork import unet_res_cfg, build_network
import matplotlib.pyplot as plt
import numpy as np
import os

os.environ["CUDA_VISIBLE_DEVICES"] = '4,5'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

n_steps = 1000

config = unet_res_cfg
net = build_network(config, n_steps)
# test DDPM
# ddpm = DDPM(device, n_steps)

# test DDIM
ddim = DDIM(device, n_steps)

static_dic = torch.load("models/model_unet_res_epoch200.pth")
net.load_state_dict(static_dic)

res = ddim.sample_backward((9, 1, 28, 28), net, device, simple_var=False, eta=0).detach().cpu()

def show_image(image, ax):
    image = image.squeeze()  # Remove single-dimensional entries from the shape
    ax.imshow(image, cmap='gray')


fig, axes = plt.subplots(3, 3, figsize=(10, 7))
axes = axes.flatten()
for i, ax in enumerate(axes):
    show_image(res[i], ax)
    ax.axis('off')  # Hide the axes

plt.tight_layout()
plt.show()