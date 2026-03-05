import torchvision
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor, Compose, Normalize, Lambda

def get_dataloader(batch_size: int):
    # 将像素取值从[0,1]调整到[-1,1]
    transform = Compose([ToTensor(), Lambda(lambda x: (x - 0.5) * 2)])
    dataset = torchvision.datasets.MNIST(root='./data',
                                         transform=transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)

def get_img_shape():
    return next(enumerate(get_dataloader(batch_size=1)))[1][0].shape[1:]