import torch
import time
from onion_model import *
from pathlib import Path

def execution_time(out_dir,device):
    model = torch.load(f'{out_dir}/train/model_best.pth', map_location=torch.device(device))
    model.eval()  # 设置模型为评估模式
    model_name = model.__class__.__name__
    print(model_name)
    if "EXPEAST" in out_dir or "phantomEAST" in out_dir:
        n = 92
        r = 50
        z = 75
    elif "EXP2A" in out_dir or "phantom2A" in out_dir:
        n = 40
        r = 32
        z = 36
    else:
        print(f"out_dir: {out_dir}中不包含'EXPEAST'或者'EXP2A'，请检查")
        exit(1)

    if  "Onion_input" in model_name or  "ResOnion_input" in model_name:
        input = torch.rand(1,n).to(device)
        start = time.time()
        print(f"start time: {start}")
        for i in range(1000):
            model(input)
        end = time.time()
        print(f"end time: {end}")
        print(f"{model_name} time consuming: {(end - start)}ms")
    elif "Onion_PI" in model_name or  "ResOnion_PI" in model_name:
        posi = torch.rand(1, n, r, z).to(device)
        input = torch.rand(1, n).to(device)
        start = time.time()
        print(f"start time: {start}")
        for i in range(1000):
            model(input, posi)
        end = time.time()
        print(f"end time: {end}")
        print(f"{model_name} time consuming: {(end - start)}ms")
    else:
        print(f"是不支持的模型")
    return end - start

if __name__ == "__main__" :
    # case_path = "../../onion_train_data/train_results_EAST/"
    case_path = "../../onion_train_data/train_results_2A/"
    # 创建Path对象
    path = Path(case_path)
    # 获取该层级的所有文件夹，不会递归到子文件夹中
    folders = [f.name for f in path.iterdir() if f.is_dir()]
    device = "cpu"
    with open(case_path+device+"_"+'execution_time_output.txt', 'a') as file:
        for name in folders:
            out_dir = case_path + name
            time_consuming_ms = execution_time(out_dir,device)
            output_string  = f"{name}: {time_consuming_ms}ms\n"
            file.write(output_string)
    print("finish")






