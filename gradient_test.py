import numpy as np
import torch

# 创建一个1x100的张量，初始设置为可计算梯度
x = np.ones((1, 100))

# 分离出前20个元素，并保持其可计算梯度
x_front = torch.tensor(x[:, :20], requires_grad=True)
# 分离出后80个元素，并设置为不计算梯度
x_back = torch.tensor(x[:, 20:], requires_grad=False)

# 将两部分重新组合成一个张量
x_split = torch.cat([x_front, x_back], dim=1)

# 定义一个简单的操作，例如乘以一个张量
y = x_split * 2

# 计算损失并反向传播
loss = y.sum()
loss.backward()

# 查看梯度是否只在前20个元素中计算
print(x_front.grad)  # 应该有梯度
print(x_back.grad)  # 应该没有梯度，因为requires_grad为False