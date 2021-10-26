import os
import torch
from PIL import Image
from torch.utils.data import DataLoader
from torchvision.transforms import transforms

from dataset import CycleDataset
from discriminator import Discriminator
from generater import Generater

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

def get_dataloader(root, image_size, batch_size, mode):
    """
    获取Data Loader，包含transform方法：
    - 重置尺寸为image_size
    - 随机裁剪
    - 随机翻转
    - 转为矩阵
    - 标准化
    """

    transform = transforms.Compose([
        transforms.Resize(int(image_size * 1.12), Image.BICUBIC),
        transforms.RandomCrop(image_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    dataset = CycleDataset(root, transform=transform, mode=mode)
    if mode == 'train':
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    elif mode == 'test':
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    return dataloader


def weights_init(m):
    """
    初始化网络参数
    """

    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        torch.nn.init.normal_(m.weight, 1.0, 0.02)
        torch.nn.init.zeros_(m.bias)


def denormalize(data):
    """
    逆标准化，将标准化后的矩阵转为原矩阵
    """

    mean, std = (.5, .5, .5), (.5, .5, .5)
    mean = torch.tensor(mean).unsqueeze(-1).unsqueeze(-1).to(device)
    std = torch.tensor(std).unsqueeze(-1).unsqueeze(-1).to(device)
    return data * std + mean

def load_models(historical_epochs, output_model_root, image_size):
    """
    读取模型
    """
    
    G_A2B = Generater().to(device)
    G_B2A = Generater().to(device)
    D_A = Discriminator(image_size).to(device)
    D_B = Discriminator(image_size).to(device)

    if historical_epochs == 0:
        G_A2B.apply(weights_init)
        G_B2A.apply(weights_init)
        D_A.apply(weights_init)
        D_B.apply(weights_init)
        return G_A2B, G_B2A, D_A ,D_B

    if historical_epochs == -1:
        paths = sorted(os.listdir(output_model_root))
        last_epoch = 0
        for path in paths:
            if path.startswith('epoch'):
                epoch = int(path[5:])
                if epoch > last_epoch:
                    last_epoch = epoch
        historical_root = os.path.join(output_model_root, f'epoch{epoch}')
    else:
        historical_root = os.path.join(output_model_root, f'epoch{historical_epochs}')

    G_A2B.load_state_dict(torch.load(os.path.join(historical_root, 'G_A2B.pth')))
    G_B2A.load_state_dict(torch.load(os.path.join(historical_root, 'G_B2A.pth')))
    D_A.load_state_dict(torch.load(os.path.join(historical_root, 'D_A.pth')))
    D_B.load_state_dict(torch.load(os.path.join(historical_root, 'D_B.pth')))

    return G_A2B, G_B2A, D_A ,D_B



