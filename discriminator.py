import torch
from torch import nn
import torch.nn.functional as F

class Discriminator(nn.Module):
    def __init__(self):
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
            nn.AvgPool2d(2),
            nn.Flatten(1),
        )
    
    def forward(self,x):
        x = self.main(x)
        return x

if __name__ == '__main__':
    D = Discriminator()
    x = torch.randn(1,3,32,32)
    print(D(x))
