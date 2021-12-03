import torch
from torch import nn
import torch.nn.functional as F


class Discriminator(nn.Module):
    """
    判别器，输入某类图片，输出图片属于该类的可能性
    [b,3,img_size,img_size] -[D]-> scaler
    """

    def __init__(self, img_size):
        super().__init__()

        self.main = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(.2, inplace=True),

            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(128),
            nn.LeakyReLU(.2, inplace=True),

            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(256),
            nn.LeakyReLU(.2, inplace=True),

            nn.Conv2d(256, 512, kernel_size=4, padding=1),
            nn.InstanceNorm2d(512),
            nn.LeakyReLU(.2, inplace=True),

            nn.Conv2d(512, 1, kernel_size=4, padding=1),
        )

        x = torch.randn(1, 3, img_size, img_size)
        x = self.main(x)
        pool_size = x.size()[2:]

        self.output = nn.Sequential(
            nn.AvgPool2d(pool_size),
            nn.Flatten(1),
        )

    def forward(self, x):
        x = self.main(x)
        x = self.output(x)
        return x


if __name__ == '__main__':
    D = Discriminator(256)
    print(D)

    x = torch.randn(1,3,256,256)
    output = D(x)
    print(output)
