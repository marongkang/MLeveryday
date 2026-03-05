import torch
import torch.nn as nn
from DDPM import DDPM
from DataLoader import get_dataloader
from DenoiseNetwork import unet_res_cfg, build_network
import numpy as np
import os

def train(ddpm: DDPM, net, device, ckpt_path, lr):
    n_steps = ddpm.n_steps
    dataloader = get_dataloader(batch_size)
    net = net.to(device)
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(net.parameters(), lr)

    for e in range(n_epochs):
        for x, _ in dataloader:
            current_batch_size = x.shape[0]
            x = x.to(device)
            t = torch.randint(0, n_steps, (current_batch_size, )).to(device)
            eps = torch.randn_like(x).to(device)
            x_t = ddpm.sample_forward(x, t, eps)
            eps_theta = net(x_t, t.reshape(current_batch_size, 1))
            loss = loss_fn(eps_theta, eps)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if e % 10 == 0:
                print(f"epoch: {e}, loss: {loss.cpu().detach().numpy()}")
        if e % 50 == 0:
            torch.save(net.state_dict(), f"{ckpt_path}model_unet_res_epoch{e}.pth")

os.environ["CUDA_VISIBLE_DEVICES"] = '2,4,7'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
n_steps = 1000

config = unet_res_cfg
net = build_network(config, n_steps)
ddpm = DDPM(device, n_steps)

lr = 1e-3
batch_size = 512
n_epochs = 1000
ckpt_path = "models/"

train(ddpm, net, device, ckpt_path, lr)
