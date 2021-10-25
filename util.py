import torch
from PIL import Image
from torch.utils.data import DataLoader
from torchvision.transforms import transforms

from dataset import CycleDataset


def get_dataloader(root, image_size, batch_size, mode):
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
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=None)
    return dataloader


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        torch.nn.init.normal_(m.weight, 1.0, 0.02)
        torch.nn.init.zeros_(m.bias)


def denormalize(data):
    mean, std = (.5, .5, .5), (.5, .5, .5)
    mean = torch.tensor(mean).unsqueeze(-1).unsqueeze(-1)
    std = torch.tensor(std).unsqueeze(-1).unsqueeze(-1)
    return data * std + mean
