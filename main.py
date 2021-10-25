import itertools
import random
import torch
import visdom
from torch.utils.data import dataloader
from tqdm import tqdm

from discriminator import Discriminator
from generater import Generater
from replay_buffer import ReplayBuffer
from util import get_dataloader, weights_init, denormalize

if torch.cuda.is_available():
    print('CUDA is available.')
    device = torch.device('cuda')
else:
    print('CUDA is not available, use CPU.')
    device = torch.device('cpu')

seed = 123
random.seed(seed)
torch.manual_seed(seed)

root = 'C:\code\cyclegan-demo\dataset\data'
image_size = 32
batch_size = 16
train_loader = get_dataloader(root,
                              image_size=image_size,
                              batch_size=batch_size,
                              mode='train')

test_loader = get_dataloader(root,
                             image_size=image_size,
                             batch_size=batch_size,
                             mode='test')

G_A2B = Generater().to(device)
G_B2A = Generater().to(device)
D_A = Discriminator().to(device)
D_B = Discriminator().to(device)

G_A2B.apply(weights_init)
G_B2A.apply(weights_init)
D_A.apply(weights_init)
D_B.apply(weights_init)

lr = 2e-4
betas = (.5, .999)
G_optim = torch.optim.Adam(itertools.chain(G_A2B.parameters(),
                                           G_B2A.parameters()),
                           lr=lr, betas=betas)
D_A_optim = torch.optim.Adam(D_A.parameters(), lr=lr, betas=betas)
D_B_optim = torch.optim.Adam(D_B.parameters(), lr=lr, betas=betas)

cycle_loss = torch.nn.L1Loss().to(device)
identity_loss = torch.nn.L1Loss().to(device)
adversarial_loss = torch.nn.MSELoss().to(device)

fake_A_buffer = ReplayBuffer()
fake_B_buffer = ReplayBuffer()

epochs = 100
save_every = 1
loss_range = 100
viz = visdom.Visdom()
g_loss, d_a_loss, d_b_loss = [], [], []
for epoch in range(epochs):
    pbar = tqdm(enumerate(train_loader), total=len(train_loader))
    G_A2B.train()
    G_B2A.train()
    D_A.train()
    D_B.train()

    for index, data in pbar:
        real_A, real_B = data['A'].to(device), data['B'].to(device)

        viz.images(denormalize(real_A), nrow=8, win='真实图片A', opts={'title': '真实图片A'})
        viz.images(denormalize(real_B), nrow=8, win='真实图片B', opts={'title': '真实图片B'})

        b_size = real_A.size(0)

        real_label = torch.full((b_size, 1), 1.).to(device)
        fake_label = torch.full((b_size, 1), 0.).to(device)

        # 更新生成器
        # 根据A、B数据集的真实图片生成B、A数据集的一致图片
        # Real_A -[G_A2B]-> Identity_B
        # Real_B -[G_B2A]-> Identity_A
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
        fake_A = G_B2A(real_B)
        fake_B = G_A2B(real_A)

        viz.images(denormalize(fake_A), nrow=8, win='虚假图片A', opts={'title': '虚假图片A'})
        viz.images(denormalize(fake_B), nrow=8, win='虚假图片B', opts={'title': '虚假图片B'})

        output_A = D_A(fake_A)
        output_B = D_B(fake_B)

        output_A_loss = adversarial_loss(output_A, real_label)
        output_B_loss = adversarial_loss(output_B, real_label)

        # 根据虚假图片重建原图
        # Fake_A -[G_A2B]-> Recovered_B
        # Fake_B -[G_B2A]-> Recovered_A
        recovered_A = G_B2A(fake_B)
        recovered_B = G_A2B(fake_A)

        viz.images(denormalize(recovered_A), nrow=8, win='重建图片A', opts={'title': '重建图片A'})
        viz.images(denormalize(recovered_B), nrow=8, win='重建图片B', opts={'title': '重建图片B'})

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
        if len(g_loss) > loss_range:
            g_loss.pop(0)
        viz.line(g_loss, win='生成器Loss', opts={'title':'生成器Loss'})

        # 更新判别器A
        real_output_A = D_A(real_A)
        real_output_A_loss = adversarial_loss(real_output_A, real_label)

        fake_A = fake_A_buffer.push_and_pop(fake_A)
        fake_output_A = D_A(fake_A.detach())
        fake_output_A_loss = adversarial_loss(fake_output_A, fake_label)

        loss = (real_output_A_loss + fake_output_A_loss) / 2
        D_A_optim.zero_grad()
        loss.backward()
        D_A_optim.step()

        d_a_loss.append(loss.item())
        if len(d_a_loss) > loss_range:
            d_a_loss.pop(0)
        viz.line(d_a_loss, win='判别器A Loss', opts={'title':'判别器A Loss'})

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
        if len(d_b_loss) > loss_range:
            d_b_loss.pop(0)
        viz.line(d_b_loss, win='判别器B Loss', opts={'title':'判别器B Loss'})

    if (epoch+1) % save_every == 0:
        torch.save(G_A2B.state_dict(),
                   f'output/model/G_A2B_epoch_{epoch+1}.pth')
        torch.save(G_B2A.state_dict(),
                   f'output/model/G_B2A_epoch_{epoch+1}.pth')
        torch.save(D_A.state_dict(), f'output/model/D_A_epoch_{epoch+1}.pth')
        torch.save(D_B.state_dict(), f'output/model/D_B_epoch_{epoch+1}.pth')
