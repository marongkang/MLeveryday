import torch
import torch.nn as nn


# MiniUnet MNIST 28*28
class DownLayer(nn.Module):
    """MiniUnet的下采样层 Resnet"""

    def __init__(self, in_channels, out_channels, time_emb_dim=16, downsample=False):
        super(DownLayer, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU()

        # 线性层，用于时间编码换通道 [B, dim] -> [B, in_channels]
        self.fc = nn.Linear(time_emb_dim, in_channels)

        self.shortcut = (
            nn.Conv2d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels
            else None
        )

        self.downsample = downsample
        if downsample:
            self.pool = nn.MaxPool2d(2)

    def forward(self, x, temb):
        res = x
        x = x + self.fc(temb)[:, :, None, None]  # [B, in_channels, 1, 1]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.act(x)

        if self.shortcut is not None:
            res = self.shortcut(res)
        x = x + res

        if self.downsample:
            x = self.pool(x)
        return x


class UpLayer(nn.Module):
    """MiniUnet的上采样层"""

    def __init__(self, in_channels, out_channels, time_emb_dim=16, upsample=False):
        super(UpLayer, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU()

        self.fc = nn.Linear(time_emb_dim, in_channels)

        self.shortcut = (
            nn.Conv2d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels
            else None
        )

        self.upsample = nn.Upsample(scale_factor=2) if upsample else None

    def forward(self, x, temb):
        if self.upsample:
            x = self.upsample(x)
        res = x
        x = x + self.fc(temb)[:, :, None, None]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.act(x)

        if self.shortcut is not None:
            res = self.shortcut(res)
        x = x + res
        return x


class MiddleLayer(nn.Module):
    """MiniUnet的中间层"""

    def __init__(self, in_channels, out_channels, time_emb_dim=16):
        super(MiddleLayer, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU()

        self.fc = nn.Linear(time_emb_dim, in_channels)

        self.shortcut = (
            nn.Conv2d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels
            else None
        )

    def forward(self, x, temb):
        res = x
        x = x + self.fc(temb)[:, :, None, None]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.act(x)

        if self.shortcut is not None:
            x = self.shortcut(x)
        x = x + res
        return x


class MiniUnet(nn.Module):
    """采用MiniUnet，对MNIST数据做生成
       两个下采样block，一个中间block，两个上采样block
    """

    def __init__(self, base_channels=16, time_emb_dim=None):
        super(MiniUnet, self).__init__()

        if time_emb_dim is None:
            self.time_emb_dim = base_channels
        else:
            self.time_emb_dim = time_emb_dim

        self.base_channels = base_channels
        self.conv_in = nn.Conv2d(1, base_channels, kernel_size=3, padding=1)

        self.down1 = nn.ModuleList([
            DownLayer(base_channels, base_channels * 2, time_emb_dim=self.time_emb_dim),
            DownLayer(base_channels * 2, base_channels * 2, time_emb_dim=self.time_emb_dim)
        ])
        self.maxpool1 = nn.MaxPool2d(2)

        self.down2 = nn.ModuleList([
            DownLayer(base_channels * 2, base_channels * 4, time_emb_dim=self.time_emb_dim),
            DownLayer(base_channels * 4, base_channels * 4, time_emb_dim=self.time_emb_dim)
        ])
        self.maxpool2 = nn.MaxPool2d(2)

        self.middle = MiddleLayer(base_channels * 4, base_channels * 4, time_emb_dim=self.time_emb_dim)

        self.upsample1 = nn.Upsample(scale_factor=2)
        self.up1 = nn.ModuleList([
            UpLayer(base_channels * 8, base_channels * 2, time_emb_dim=self.time_emb_dim),
            UpLayer(base_channels * 2, base_channels * 2, time_emb_dim=self.time_emb_dim)
        ])
        self.upsample2 = nn.Upsample(scale_factor=2)
        self.up2 = nn.ModuleList([
            UpLayer(base_channels * 4, base_channels, time_emb_dim=self.time_emb_dim),
            UpLayer(base_channels, base_channels, time_emb_dim=self.time_emb_dim)
        ])

        self.conv_out = nn.Conv2d(base_channels, 1, kernel_size=1, padding=0)

    def time_emb(self, t, dim):
        """正弦时间编码"""
        t = t * 1000
        freqs = torch.pow(10000, torch.linspace(0, 1, dim // 2)).to(t.device)
        sin_emb = torch.sin(t[:, None] / freqs)
        cos_emb = torch.cos(t[:, None] / freqs)
        return torch.cat([sin_emb, cos_emb], dim=-1)

    def label_emb(self, y, dim):
        """正弦时间编码"""
        y = y * 1000
        freqs = torch.pow(10000, torch.linspace(0, 1, dim // 2)).to(y.device)
        sin_emb = torch.sin(y[:, None] / freqs)
        cos_emb = torch.cos(y[:, None] / freqs)
        return torch.cat([sin_emb, cos_emb], dim=-1)

    def forward(self, x, t, y=None):
        """仅基于时间编码的前向传播"""
        x = self.conv_in(x)
        temb = self.time_emb(t, self.base_channels)

        if y is not None:
            # 判断y是label还是token
            if len(y.shape) == 1:
                # label编码，-1表示无条件生成，仅用于训练区分，推理的时候不需要
                # 把y中等于-1的部分找出来不进行任何编码，其余的进行编码
                yemb = self.label_emb(y, self.base_channels)
                # 把y等于-1的index找出来，然后把对应的y_emb设置为0
                yemb[y == -1] = 0.0
                temb += yemb
            else:  # 文字版本
                pass

        for layer in self.down1:
            x = layer(x, temb)
        x1 = x
        x = self.maxpool1(x)
        for layer in self.down2:
            x = layer(x, temb)
        x2 = x
        x = self.maxpool2(x)

        x = self.middle(x, temb)

        x = torch.cat([self.upsample1(x), x2], dim=1)
        for layer in self.up1:
            x = layer(x, temb)
        x = torch.cat([self.upsample2(x), x1], dim=1)
        for layer in self.up2:
            x = layer(x, temb)

        x = self.conv_out(x)
        return x


if __name__ == '__main__':
    device = 'cuda'
    model = MiniUnet().to(device)
    x = torch.randn(128, 1, 28, 28).to(device)
    t = torch.randn(128).to(device)

    out = model(x, t)
    print(out.shape)
