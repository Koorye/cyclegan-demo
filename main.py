import itertools
import random
import os
import torch
import visdom
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm

from replay_buffer import ReplayBuffer
from test import generate_images
from util import get_dataloader, denormalize, load_models

# 初始化参数
# seed: 随机种子
# root: 数据集路径
# output_model_root: 模型的输出路径
# image_size: 图片尺寸
# batch_size: 一次喂入的数据量
# lr: 学习率
# betas: 一阶和二阶动量
# epochs: 训练总次数
# historical_epochs: 历史训练次数
# - 0表示不沿用历史模型
# - >0表示对应训练次数的模型
# - -1表示最后一次训练的模型
# decay_epoch: 从第几次训练开始衰减学习率
# save_every: 保存频率
# loss_range: Loss的显示范围
seed = 123
data_root = 'data/summer2winter'
output_model_root = 'output/model'
output_img_root = 'output/img'
image_size = 256
batch_size = 1
lr = 2e-4
betas = (.5, .999)
epochs = 200
historical_epochs = 0
decay_epoch = 3
save_every = 5
loss_range = 1000

# 创建输出目录
if not os.path.exists(output_model_root):
    os.makedirs(output_model_root)
if not os.path.exists(output_img_root):
    os.makedirs(output_img_root)

# 检测CUDA是否可用
print('=========================================')
if torch.cuda.is_available():
    print('CUDA已启用.')
    device = torch.device('cuda')
else:
    print('CUDA不可用, 使用CPU.')
    device = torch.device('cpu')

# 初始化随机种子
print(f'初始化种子{seed}')
random.seed(seed)
torch.manual_seed(seed)

# 读取Data Loader
print(f'读取数据，路径为{data_root}，尺寸为{image_size}x{image_size}，每次喂入{batch_size}')
train_loader = get_dataloader(data_root,
                              image_size=image_size,
                              batch_size=batch_size,
                              mode='train')

# 读取历史模型
print('读取模型、优化器、误差')
G_A2B, G_B2A, D_A, D_B, last_epoch = load_models(
    historical_epochs, output_model_root, image_size)

# 初始化优化器
G_optim = torch.optim.Adam(itertools.chain(G_A2B.parameters(),
                                           G_B2A.parameters()),
                           lr=lr, betas=betas)
D_A_optim = torch.optim.Adam(D_A.parameters(), lr=lr, betas=betas)
D_B_optim = torch.optim.Adam(D_B.parameters(), lr=lr, betas=betas)

# 初始化学习率调度器
def lr_lambda(ep):
    return 1.0 - max(0, ep + last_epoch - decay_epoch) / (epochs - decay_epoch)

G_lr = LambdaLR(G_optim, lr_lambda=lr_lambda)
D_A_lr = LambdaLR(D_A_optim, lr_lambda=lr_lambda)
D_B_lr = LambdaLR(D_B_optim, lr_lambda=lr_lambda)

# 初始化误差
cycle_loss = torch.nn.L1Loss().to(device)
identity_loss = torch.nn.L1Loss().to(device)
adversarial_loss = torch.nn.MSELoss().to(device)

# 缓冲区用于存放虚假图片
fake_A_buffer = ReplayBuffer()
fake_B_buffer = ReplayBuffer()

print('加载Visdom...')
viz = visdom.Visdom()
g_loss, d_a_loss, d_b_loss = [], [], []
print('=========================================')
for epoch in range(epochs-last_epoch):
    epoch += last_epoch
    pbar = tqdm(enumerate(train_loader), total=len(
        train_loader), desc=f'第{epoch+1}次训练')
    G_A2B.train()
    G_B2A.train()
    D_A.train()
    D_B.train()

    for index, data in pbar:
        real_A, real_B = data['A'].to(device), data['B'].to(device)

        viz.images(denormalize(real_A), nrow=8,
                   win='真实图片A', opts={'title': '真实图片A'})
        viz.images(denormalize(real_B), nrow=8,
                   win='真实图片B', opts={'title': '真实图片B'})

        b_size = real_A.size(0)

        real_label = torch.full((b_size, 1), 1.).to(device)
        fake_label = torch.full((b_size, 1), 0.).to(device)

        # 更新生成器
        # 根据A、B数据集的真实图片生成B、A数据集的一致图片
        # Real_A -[G_A2B]-> Identity_B
        # Real_B -[G_B2A]-> Identity_A
        # 计算一致性损失，权重为5
        # L1Loss(原图片 -> 生成图片，原图片)
        identity_A = G_B2A(real_B)
        identity_B = G_A2B(real_A)

        identity_A_loss = identity_loss(identity_A, real_A) * 5.0
        identity_B_loss = identity_loss(identity_B, real_B) * 5.0

        # 根据A、B数据集的真实图片生成B、A数据集的虚假图片
        # Real_A -[G_A2B]-> Fake_B
        # Real_B -[G_B2A]-> Fake_A
        # 根据虚假图片进行判别
        # Fake_A -[D_A]-> Label_A
        # Fake_B -[D_B]-> Label_B
        # 计算对抗损失，权重为1
        # MSELoss(原图片 -> 生成图片 -> 判别结果，1)
        fake_A = G_B2A(real_B)
        fake_B = G_A2B(real_A)

        viz.images(denormalize(fake_A), nrow=8,
                   win='虚假图片A', opts={'title': '虚假图片A'})
        viz.images(denormalize(fake_B), nrow=8,
                   win='虚假图片B', opts={'title': '虚假图片B'})

        output_A = D_A(fake_A)
        output_B = D_B(fake_B)

        output_A_loss = adversarial_loss(output_A, real_label)
        output_B_loss = adversarial_loss(output_B, real_label)

        # 根据虚假图片重建原图
        # Fake_A -[G_A2B]-> Recovered_B
        # Fake_B -[G_B2A]-> Recovered_A
        # 计算循环损失，权重为10
        # L1Loss(原图片 -> 生成图片 -> 重建图片, 原图片)
        recovered_A = G_B2A(fake_B)
        recovered_B = G_A2B(fake_A)

        viz.images(denormalize(recovered_A), nrow=8,
                   win='重建图片A', opts={'title': '重建图片A'})
        viz.images(denormalize(recovered_B), nrow=8,
                   win='重建图片B', opts={'title': '重建图片B'})

        recovered_A_loss = cycle_loss(recovered_A, real_A) * 10.0
        recovered_B_loss = cycle_loss(recovered_B, real_B) * 10.0

        # 更新梯度
        loss = (identity_A_loss + identity_B_loss
                + output_A_loss + output_B_loss
                + recovered_A_loss + recovered_B_loss)
        G_optim.zero_grad()
        loss.backward()
        G_optim.step()

        g_loss.append(loss.item())
        if loss_range != -1 and len(g_loss) > loss_range:
            g_loss.pop(0)
        viz.line(g_loss, win='生成器Loss', opts={'title': '生成器Loss'})

        # 更新判别器A
        # 计算对抗损失
        # MSELoss(A类真实图片 -> A类判别结果, 1)
        real_output_A = D_A(real_A)
        real_output_A_loss = adversarial_loss(real_output_A, real_label)

        # 计算对抗损失
        # MSELoss(A类生成图片 -> A类判别结果, 0)
        fake_A = fake_A_buffer.push_and_pop(fake_A)
        fake_output_A = D_A(fake_A.detach())
        fake_output_A_loss = adversarial_loss(fake_output_A, fake_label)

        loss = (real_output_A_loss + fake_output_A_loss) / 2
        D_A_optim.zero_grad()
        loss.backward()
        D_A_optim.step()

        d_a_loss.append(loss.item())
        if loss_range != -1 and len(d_a_loss) > loss_range:
            d_a_loss.pop(0)
        viz.line(d_a_loss, win='判别器A Loss', opts={'title': '判别器A Loss'})

        # 更新判别器B
        real_output_B = D_B(real_B)
        real_output_B_loss = adversarial_loss(real_output_B, real_label)

        fake_B = fake_B_buffer.push_and_pop(fake_B)
        fake_output_B = D_B(fake_B.detach())
        fake_output_B_loss = adversarial_loss(fake_output_B, fake_label)

        loss = (real_output_B_loss + fake_output_B_loss) / 2
        D_B_optim.zero_grad()
        loss.backward()
        D_B_optim.step()

        d_b_loss.append(loss.item())
        if loss_range != -1 and len(d_b_loss) > loss_range:
            d_b_loss.pop(0)
        viz.line(d_b_loss, win='判别器B Loss', opts={'title': '判别器B Loss'})
    
    G_lr.step()
    D_A_lr.step()
    D_B_lr.step()

    # 生成图片和保存模型
    if (epoch+1) % save_every == 0:
        A2B_img, B2A_img = generate_images(G_A2B, G_B2A)
        A2B_img.save(os.path.join(output_img_root, f'A2B_epoch{epoch+1}.png'))
        B2A_img.save(os.path.join(output_img_root, f'B2A_epoch{epoch+1}.png'))

        output_model_epoch_root = os.path.join(
            output_model_root, f'epoch{epoch+1}')
        os.mkdir(output_model_epoch_root)
        torch.save(G_A2B.state_dict(), os.path.join(
            output_model_epoch_root, 'G_A2B.pth'))
        torch.save(G_B2A.state_dict(), os.path.join(
            output_model_epoch_root, 'G_B2A.pth'))
        torch.save(D_A.state_dict(), os.path.join(
            output_model_epoch_root, 'D_A.pth'))
        torch.save(D_B.state_dict(), os.path.join(
            output_model_epoch_root, 'D_B.pth'))
        print(f'训练完成，模型以保存至{output_model_epoch_root}')
