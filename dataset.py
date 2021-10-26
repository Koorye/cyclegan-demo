import glob
import matplotlib.pyplot as plt
import os
from PIL import Image
from torch.utils.data import Dataset


class CycleDataset(Dataset):
    """
    用于CycleGAN的数据集
    分别读取A和B的目录，目录中包含两类图片
    每个获取元素，将返回一个字典 {'A': A_Image, 'B': B_Image}
    """

    def __init__(self, root, transform=None, mode='train'):
        self.transform = transform

        a_dir = os.path.join(root, f'{mode}A')
        b_dir = os.path.join(root, f'{mode}B')

        self.a_files = sorted(glob.glob(a_dir+'/*.*'))
        self.b_files = sorted(glob.glob(b_dir+'/*.*'))

    def __getitem__(self, index):
        if self.transform is None:
            a = Image.open(self.a_files[index % len(self.a_files)])
            b = Image.open(self.b_files[index % len(self.b_files)])
        else:
            a = self.transform(Image.open(
                self.a_files[index % len(self.a_files)]))
            b = self.transform(Image.open(
                self.b_files[index % len(self.b_files)]))

        return {
            'A': a,
            'B': b
        }

    def __len__(self):
        return max(len(self.a_files), len(self.b_files))

if __name__ == '__main__':
    root = 'C:\code\cyclegan-demo\dataset\data'
    dataset = CycleDataset(root, None, 'train')
    item_dict = dataset.__getitem__(123)
    a_img, b_img = item_dict['A'], item_dict['B']

    plt.clf()
    plt.imshow(a_img)
    plt.show()

    plt.clf()
    plt.imshow(b_img)
    plt.show()
    
