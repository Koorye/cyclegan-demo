import math
import matplotlib.pyplot as plt
import numpy as np
import os
from PIL import Image
import torch

from util import load_models, get_dataloader, denormalize

historical_epochs = -1
batch_size = 16
image_size = 64
data_root = 'data/summer2winter'
output_model_root = 'output/model'
output_img_root = 'output/img'

if not os.path.exists(output_img_root):
    os.makedirs(output_img_root)

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

G_A2B, G_B2A, D_A, D_B = load_models(
    historical_epochs, output_model_root, image_size)

test_loader = get_dataloader(data_root, image_size, batch_size, 'test')
test_data = next(iter(test_loader))
A_data, B_data = test_data['A'].to(device), test_data['B'].to(device)

output_A_data = G_B2A(B_data)
output_B_data = G_A2B(A_data)


def draw_images(data, ncol):
    with torch.no_grad():
        b, c, w, h = data.size(0), data.size(1), data.size(2), data.size(3)
        data = denormalize(data)
        data = data.transpose(1, 3).transpose(1, 2).cpu().numpy() * 255

        nrow = math.ceil(b / ncol)
        plot = np.zeros((h*nrow, w*ncol, c)).astype(np.uint8)

        row, col = 0, 0
        for img in data:
            plot[row*h:(row+1)*h, col*w:(col+1)*w, :] = img
            col += 1
            if col == ncol:
                row, col = row+1, 0
    return Image.fromarray(plot)


def concat_images(imgs, ncol):
    shape = np.array(imgs[0]).shape
    w, h = shape[0], shape[1]

    nrow = math.ceil(len(imgs) / ncol)
    target_shape = (ncol * w, nrow * h)
    target = Image.new('RGB', target_shape)


    row, col = 0, 0
    for img in imgs:
        target.paste(img, (col*w, row*h, (col+1)*w, (row+1)*h))
        col += 1
        if col == ncol:
            row, col = row+1, 0
    return target

real_A_imgs = draw_images(A_data, 4)
real_B_imgs = draw_images(B_data, 4)
output_A_imgs = draw_images(output_A_data, 4)
output_B_imgs = draw_images(output_B_data, 4)

A2B_img = concat_images([real_A_imgs, output_B_imgs], 2)
B2A_img = concat_images([real_B_imgs, output_A_imgs], 2)
A2B_img.save(os.path.join(output_img_root, 'A2B.png'))
B2A_img.save(os.path.join(output_img_root, 'B2A.png'))