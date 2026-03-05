import math
import numpy as np

import torch
import torch.nn.functional as F

import transformer.Constants as Constants
from transformer.Models import Transformer
from transformer.Optim import ScheduledOptim


transformer = Transformer(
    n_src_vocab = 12,
    n_trg_vocab = 12,
    src_pad_idx = 0,
    trg_pad_idx = 0,
).to('cuda')

input_seq = torch.ones([2,10], dtype=torch.long).to('cuda')
target_seq = torch.ones([2,10], dtype=torch.long).to('cuda')
target_seq = transformer(input_seq, target_seq)

print(target_seq.shape)