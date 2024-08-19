import os
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from train import get_dataset
from model import Onion, Config, region_embedding_func, position_embedding_func
from torch.utils.data import DataLoader
import torch.nn as nn
import time

def predict(config, model_load_path, model_name):
    model = Onion(config)
    model.load_state_dict(torch.load(model_load_path, map_location=torch.device('cpu')))
    model = model.to(device)

    model.eval()
    preds, labels = [], []
    test_loss = []
    t1 = time.time()
    for batch in test_iter:
        with torch.no_grad():
            batch = [el.cuda() for el in batch]
            outp = model(batch[0])
            loss = loss_MSE(outp, batch[1])
            preds.append(outp)
            labels.append(batch[1])
        test_loss.append(loss.item())
    t2 = time.time()
    test_loss = sum(test_loss) / len(test_loss)
    print(f'test set loss is: {test_loss}, test time is: {t2 - t1} s')

    preds = preds.cpu().numpy()
    labels = labels.cpu().numpy()
    mae = np.mean(np.abs(preds - labels))
    print(f'test set mae = np.mean(np.abs(preds - labels)): {mae}')
    df_preds = pd.DataFrame(preds)
    df_labels = pd.DataFrame(labels)
    # 循环结束后，将所有输出写入TXT文件
    if not os.path.exists("./outputs"):
        os.mkdir("./outputs")
    if not os.path.exists("./outputs/" + model_name):
        os.mkdir("./outputs/" + model_name)

    df_preds.to_csv("./outputs/{}/pre_matrix_.csv".format(model_name))
    df_labels.to_csv("./outputs/{}/gt_matrix_.csv".format(model_name))
    print('saved in output file')

if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # test_data = get_dataset("./model_data/cut100/valid_input.h5", "./model_data/cut100/valid_output.h5")
    test_data = get_dataset("./data_HL_2A/test_database.h5")
    test_iter = DataLoader(test_data, shuffle=False, batch_size=64, drop_last=False)
    loss_MSE = nn.MSELoss()
    model_name = "Onion_0"
    modelPath = "./model_data/Onion.../" + model_name + ".pth"
    config = Config(1024, 40, 6, 12, 36 * 32, 0.0, True, torch.float64, 128)
    predict(config = config, model_load_path=modelPath, model_name=model_name)
