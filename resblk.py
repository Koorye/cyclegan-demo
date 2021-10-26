from torch import nn


class Resblk(nn.Module):
    """
    残差块单元
    """

    def __init__(self, in_channels):
        super(Resblk, self).__init__()

        self.main = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channels, in_channels, 3),
            nn.InstanceNorm2d(in_channels),
            nn.ReLU(inplace=True),

            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channels, in_channels, 3),
            nn.InstanceNorm2d(in_channels),
        )

    def forward(self, x):
        return x + self.main(x)


if __name__ == '__main__':
    R = Resblk(3)
    print(R)
