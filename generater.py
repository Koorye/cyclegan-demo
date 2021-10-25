import torch
from torch import nn
from resblk import Resblk


class Generater(nn.Module):
    def __init__(self):
        super(Generater, self).__init__()

        # 输入层卷积
        self.input = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(3, 64, kernel_size=7),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True),
        )

        # 下采样
        self.downsampling = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm2d(256),
            nn.ReLU(inplace=True),
        )

        # 9个残差块组成残差网络
        self.resnet = nn.Sequential(
            Resblk(256),
            Resblk(256),
            Resblk(256),
            Resblk(256),
            Resblk(256),
            Resblk(256),
            Resblk(256),
            Resblk(256),
            Resblk(256),
        )

        # 上采样
        self.upsampling = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=3,
                               stride=2, padding=1, output_padding=1),
            nn.InstanceNorm2d(128),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(128, 64, kernel_size=3,
                               stride=2, padding=1, output_padding=1),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True),
        )

        # 输出层
        self.output = nn.Sequential(
            nn.ReplicationPad2d(3),
            nn.Conv2d(64, 3, 7),
            nn.Tanh(),
        )

    def forward(self, x):
        x = self.input(x)
        x = self.downsampling(x)
        x = self.resnet(x)
        x = self.upsampling(x)
        x = self.output(x)
        return x


if __name__ == '__main__':
    G = Generater()
    x = torch.randn(1, 3, 64, 64)
    print(G(x).size())
