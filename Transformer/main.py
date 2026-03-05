import torch
import torch.nn as nn


q = torch.randn((10,4,5,6))
k = torch.randn((10,4,5,6))

x = torch.randn((5,6))
y = torch.rand((6,7))
z = x@y

print(z.shape)

# print(k.transpose(-2, -1).shape)
# attn = (q @ k.transpose(-2, -1))
# print(attn.shape)