from models.swin_transformer import SwinTransformer
import torch
import os

old_repr = torch.Tensor.__repr__
def tensor_info(tensor):
    return repr(tensor.shape)[6:] + ' ' + repr(tensor.dtype)[6:] + '@' + str(tensor.device) + '\n' + old_repr(tensor)
torch.Tensor.__repr__ = tensor_info

print(torch.__version__, torch.cuda.is_available())

os.environ["CUDA_VISIBLE_DEVICES"] = "5"

f_map = torch.randn((2, 3, 224, 224)).to('cuda')
swin_tf = SwinTransformer().to('cuda')

output = swin_tf(f_map).to('cpu')

print(output.shape)