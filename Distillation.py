# coding: utf-8
# Author: yuxiang Zeng

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import *


# 定义教师模型
class TeacherModel(nn.Module):
    def __init__(self):
        super(TeacherModel, self).__init__()
        self.fc1 = nn.Linear(784, 1024)
        self.fc2 = nn.Linear(1024, 1024)
        self.fc3 = nn.Linear(1024, 10)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# 定义学生模型
class StudentModel(nn.Module):
    def __init__(self):
        super(StudentModel, self).__init__()
        self.fc1 = nn.Linear(784, 256)
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# 定义蒸馏损失函数
def distillation_loss(student_logits, labels, teacher_logits, temperature, alpha):
    soft_loss = nn.KLDivLoss(reduction='batchmean')(F.log_softmax(student_logits / temperature, dim=1),
                                                     F.softmax(teacher_logits / temperature, dim=1)) * (alpha * temperature * temperature)
    hard_loss = F.cross_entropy(student_logits, labels) * (1. - alpha)
    return soft_loss + hard_loss


if __name__ == '__main__':
    epochs = 3
    batch_size = 32
    temperature = 2.0  # 温度参数
    alpha = 0.5  # 平衡硬标签和软标签的重要性
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),  # MNIST 数据集的全局平均值和标准差
    ])
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

    teacher_model = TeacherModel()
    student_model = StudentModel()
    optimizer = optim.Adam(student_model.parameters(), lr=0.001)

    # 假设教师模型已经预训练好了，这里仅演示蒸馏过程
    # 训练学生模型
    student_model.train()
    teacher_model.eval()  # 确保教师模型处于评估模式
    for epoch in range(epochs):
        total_loss = 0.0
        for (data, target) in tqdm(train_loader):
            data, target = data.view(data.size(0), -1), target  # 调整数据形状适应全连接层的输入
            optimizer.zero_grad()
            student_output = student_model(data)
            teacher_output = teacher_model(data)
            loss = distillation_loss(student_output, target, teacher_output, temperature, alpha)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print('Train Epoch: Average Loss: {:.6f}'.format(total_loss / len(train_loader.dataset)))

