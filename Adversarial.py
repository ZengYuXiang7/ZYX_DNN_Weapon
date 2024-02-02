# coding : utf-8
# Author : yuxiang Zeng

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F


class Generator(nn.Module):
    def __init__(self, input_size, hidden_dim, output_size):
        super(Generator, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim * 2)
        self.fc3 = nn.Linear(hidden_dim * 2, hidden_dim * 4)
        self.fc4 = nn.Linear(hidden_dim * 4, output_size)

    def forward(self, x):
        x = F.leaky_relu(self.fc1(x), 0.2)
        x = F.leaky_relu(self.fc2(x), 0.2)
        x = F.leaky_relu(self.fc3(x), 0.2)
        x = torch.tanh(self.fc4(x))
        return x


class Discriminator(nn.Module):
    def __init__(self, input_size, hidden_dim, output_size):
        super(Discriminator, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_dim * 4)
        self.fc2 = nn.Linear(hidden_dim * 4, hidden_dim * 2)
        self.fc3 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, output_size)

    def forward(self, x):
        x = F.leaky_relu(self.fc1(x), 0.2)
        x = F.dropout(x, 0.3)
        x = F.leaky_relu(self.fc2(x), 0.2)
        x = F.dropout(x, 0.3)
        x = F.leaky_relu(self.fc3(x), 0.2)
        x = F.dropout(x, 0.3)
        x = torch.sigmoid(self.fc4(x))
        return x


if __name__ == '__main__':
    # 参数设置
    input_size = 100  # 生成器输入噪声的维度
    hidden_size = 256  # 内部隐藏层的维度
    output_size = 28 ** 2  # 生成的图像维度（对于MNIST为28x28=784）
    batch_size = 64  # 批处理大小
    epochs = 5

    # 实例化模型
    generator = Generator(input_size, hidden_size, output_size)
    discriminator = Discriminator(output_size, hidden_size, 1)

    # 优化器
    g_optimizer = optim.AdamW(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    d_optimizer = optim.AdamW(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

    # 损失函数
    criterion = nn.BCELoss()

    # 准备数据集
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])

    # 加载训练集
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

    # 加载测试集
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

    # 训练过程
    for epoch in range(epochs):
        for i, (images, _) in enumerate(train_loader):
            images = images.view(-1, 28 * 28)  # 展平图像
            real_labels = torch.ones(images.size(0), 1)
            fake_labels = torch.zeros(images.size(0), 1)

            # 训练判别器
            d_optimizer.zero_grad()
            pred_label = discriminator(images)
            d_loss_real = criterion(pred_label, real_labels)
            d_loss_real.backward()

            z = torch.randn(images.size(0), input_size)
            fake_images = generator(z)
            pred_label = discriminator(fake_images.detach())
            d_loss_fake = criterion(pred_label, fake_labels)
            d_loss_fake.backward()

            d_optimizer.step()
            d_loss = d_loss_real + d_loss_fake

            # 训练生成器
            g_optimizer.zero_grad()
            pred_label = discriminator(fake_images)
            g_loss = criterion(pred_label, real_labels)
            g_loss.backward()
            g_optimizer.step()

            if (i + 1) % 100 == 0:
                print(f'Epoch [{epoch + 1}/{epochs}], Step [{i + 1}/{len(train_loader)}], D Loss: {d_loss.item():.4f}, G Loss: {g_loss.item():.4f}')

    # 使用训练好的生成器生成图像
    with torch.no_grad():
        fixed_noise = torch.randn(64, input_size)
        fake_images = generator(fixed_noise).reshape(-1, 1, 28, 28)
        fake_images = (fake_images + 1) / 2  # 调整像素值到[0, 1]
        grid = vutils.make_grid(fake_images, nrow=8, padding=2, normalize=True)
        plt.figure(figsize=(10, 10))
        plt.imshow(np.transpose(grid.cpu().numpy(), (1, 2, 0)))
        plt.show()
