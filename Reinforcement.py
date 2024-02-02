import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random


# 定义环境
class GridWorld:
    def __init__(self):
        self.size = 5
        self.state = (0, 0)  # 初始状态
        self.goal = (4, 4)  # 目标状态

    def reset(self):
        self.state = (0, 0)
        return self.state

    def step(self, action):
        y, x = self.state
        if action == 0:  # 上
            y = max(y - 1, 0)
        elif action == 1:  # 下
            y = min(y + 1, self.size - 1)
        elif action == 2:  # 左
            x = max(x - 1, 0)
        elif action == 3:  # 右
            x = min(x + 1, self.size - 1)

        self.state = (y, x)
        reward = 1 if self.state == self.goal else -1 if self.state == (0, 0) else 0
        return self.state, reward


# 定义智能体网络
class QNetwork(nn.Module):
    def __init__(self):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(2, 10)  # 输入层（状态维度为2）到隐藏层
        self.fc2 = nn.Linear(10, 4)  # 隐藏层到输出层（动作空间维度为4）

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# 训练智能体

if __name__ == '__main__':
    episodes = 5000
    env = GridWorld()
    q_network = QNetwork()
    optimizer = optim.Adam(q_network.parameters(), lr=0.01)
    loss_fn = nn.MSELoss()

    for episode in range(episodes):
        state = env.reset()
        total_reward = 0

        for step in range(50):  # 每个回合最多50步
            state_tensor = torch.tensor([state], dtype=torch.float32)
            q_values = q_network(state_tensor)
            action = torch.argmax(q_values).item()

            next_state, reward = env.step(action)
            next_state_tensor = torch.tensor([next_state], dtype=torch.float32)
            next_q_values = q_network(next_state_tensor)
            max_next_q_value = torch.max(next_q_values).item()

            target_q_value = reward + (0.99 * max_next_q_value)
            target_q_values = q_values.clone().detach()
            target_q_values[0, action] = target_q_value

            loss = loss_fn(q_values, target_q_values)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            state = next_state
            total_reward += reward

            if state == env.goal:
                break

        print(f"Episode {episode + 1}, Total Reward: {total_reward}")


