import torch
import time
from onion_model import *


out_dir = "../../onion_train_data/train_results_EAST/EXPEAST_42_Onion_input_adam"
model = torch.load(f'{out_dir}/train/model_best.pth', map_location=torch.device('cuda'))
model.eval()  # 设置模型为评估模式
model_name = model.__class__.__name__

if "EXPEAST" in out_dir:
    n = 92
    r = 75
    z = 50
elif "EXP2A" in out_dir:
    n = 40
    r = 36
    z = 32
else:
    print(f"out_dir: {out_dir}中不包含'EXPEAST'或者'EXP2A'，请检查")
    exit(1)

if model_name == "Onion_input" or model_name == "ResOnion_input":
    input = torch.rand(1,n).to('cuda')
    start = time.time()
    print(f"start time: {start}")
    model(input)
    end = time.time()
    print(f"end time: {end}")
    print(f"time consuming: {end - start}")
elif model_name == "Onion_PI" or model_name == "ResOnion_PI":
    posi = torch.rand(1, n, r, z)
    input = torch.rand(1, n).to('cuda')
    start = time.time()
    print(f"start time: {start}")
    model(input)
    end = time.time()
    print(f"end time: {end}")
    print(f"time consuming: {end - start}s")
else:
    print(f"仅支持Onion_input ResOnion_input Onion_PI ResOnion_PI四个模型")

