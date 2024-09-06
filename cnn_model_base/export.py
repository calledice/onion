from dataset import OnionDataset
from torch.utils.data import Dataset, DataLoader
from onion_model import *
import torch.nn as nn
import torch
from tqdm import tqdm
import os
import json
from onion_model import Onion
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