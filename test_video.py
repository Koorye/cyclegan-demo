import cv2
import numpy as np
import os
import torch
from tqdm import tqdm

from torchvision.transforms import transforms
from util import load_models, denormalize

historical_epochs = -1
batch_size = 16
image_size = 64
output_model_root = 'output/model'
output_img_root = 'output/img'

if not os.path.exists(output_img_root):
    os.makedirs(output_img_root)

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

G_A2B, G_B2A, _, _, _ = load_models(
    historical_epochs, output_model_root, image_size)



def movie2moive(movie_path, scale=1, A2B=True, lim=-1):
    """
    : param movie_path: 输入的视频路径
    : param scale: 缩放比例，太大的尺寸可能导致显存不足
    : param A2B: A转B或B转A
    : param lim: 截取上限帧数，设为-1表示截取完整视频
    """

    moive_name = movie_path.split('/')[-1].split('.')[0]

    with torch.no_grad():
        if A2B:
            G_B2A.cpu()
        else:
            G_A2B.cpu()
        torch.cuda.empty_cache()

        video = cv2.VideoCapture(movie_path)
        w=int(video.get(cv2.CAP_PROP_FRAME_WIDTH) * scale)
        h=int(video.get(cv2.CAP_PROP_FRAME_HEIGHT) * scale)
        fps = video.get(cv2.CAP_PROP_FPS)
        fourcc = cv2.VideoWriter_fourcc(*'X264')
        video_len = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

        trans = transforms.Compose([
                transforms.ToPILImage(mode='RGB'),
                transforms.Resize((h,w)),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ])

        if not video.isOpened():
            return

        if lim >= 0:
            video_len = lim

        _, img = video.read()
        h_out, w_out = G_A2B(trans(img).to(device).unsqueeze(0)).size()[2:]
        output_video = cv2.VideoWriter(os.path.join(output_img_root, f'{moive_name}.mkv'), fourcc, fps, (w_out,h_out))

        index = 0
        pbar = tqdm(total=video_len)
        while True:
            # 帧数提取阶段
            ret, img = video.read()
            if not ret:
                break
                    
            # BGR -> GRB
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            index += 1
            if index == lim:
                break

            # 风格转换阶段
            if A2B:
                output = G_A2B(trans(img).to(device).unsqueeze(0))
            else:
                output = G_B2A(trans(img).to(device).unsqueeze(0))

            # [c,h,w] -> [h,w,c]
            output = (denormalize(output.squeeze(0)).permute(1,2,0).cpu().numpy() * 255).astype(np.uint8)
            # RGB -> BGR
            output = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)
            output_video.write(output)

            pbar.update(1)

    video.release()
    output_video.release()
        

if __name__ == '__main__':
    movie2moive('data/japan_town.mp4', A2B=True, scale=1, lim=-1)


    