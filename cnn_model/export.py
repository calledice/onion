import torch
from onion_model import CNN_Base, Onion, OnionWithoutRegi

# torch.Size([64, 23]) torch.Size([64, 32, 36]) torch.Size([64, 23, 32, 36])
input = torch.rand(64, 23, dtype=torch.float32)
regi = torch.rand(64, 32, 36, dtype=torch.float32)
posi = torch.rand(64, 23, 32, 36, dtype=torch.float32)
print(input.shape, regi.shape, posi.shape)
model = torch.load('./output/train/model_best.pth', map_location='cpu')
# torch.onnx.export(model, (input, regi, posi), 'model.onnx')
traced_model = torch.jit.trace(model, (input, regi, posi))
traced_model.save('model.jit')
# model(input, regi, posi)