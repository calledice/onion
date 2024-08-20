import torch

# 假设有一个 m x n 的 Tensor，只有 r x z 部分需要计算梯度
m, n = 5, 7
r, z = 3, 4

# 创建整个 m x n 的张量
full_tensor = torch.randn(m, n)

# 需要梯度计算的部分
requires_grad_tensor = full_tensor[:r, :z].clone().detach().requires_grad_(True)

# 不需要梯度计算的部分
no_grad_tensor = full_tensor.clone().detach()
no_grad_tensor[:r, :z] = requires_grad_tensor

# 定义一个简单的损失函数，只对需要计算梯度的部分操作
loss = no_grad_tensor.sum()
loss.backward()

# 打印梯度，观察只有 r x z 部分有梯度
print("Grad for requires_grad_tensor:")
print(requires_grad_tensor.grad)

print("\nNo grad tensor:")
print(no_grad_tensor)

# 更新只对需要梯度计算的部分
with torch.no_grad():
    requires_grad_tensor -= 0.01 * requires_grad_tensor.grad
    no_grad_tensor[:r, :z] = requires_grad_tensor

print("\nUpdated no_grad_tensor:")
print(no_grad_tensor)
