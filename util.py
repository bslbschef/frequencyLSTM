import torch
import sys
import pandas as pd
import numpy as np
from parameter import *
import os
import random


# 加载数据函数（跳过头部字符串行）
def load_data(file_path):
    data = pd.read_csv(file_path, skiprows=1, header=None).values
    return torch.tensor(data, dtype=torch.float32).unsqueeze(-1)


def vector_transfer(batch_size, window_size, inputs_file, targets_file):
    if inputs_file.size(0) == batch_size + (window_size - 1):
        inputs_file = inputs_file.permute(2, 0, 1)
        input_tensor = [inputs_file[:, i:i + window_size, :].squeeze(0) for i in
                        range(inputs_file.size(1) - window_size + 1)]
        input_tensor = torch.stack(input_tensor)

        left_index = int((window_size - 1) / 2)
        right_index = int(targets_file.size(0) - (window_size - 1) / 2)
        label_tensor = targets_file[left_index:right_index, :, :].squeeze(-1)
        return input_tensor, label_tensor
    else:
        return np.array([[0, 0]]), np.array([[0, 0]])


def vector_transfer_test(window_size, inputs_file):
    inputs_file = inputs_file.permute(2, 0, 1)
    input_tensor = [inputs_file[:, i:i + window_size, :].squeeze(0) for i in
                    range(inputs_file.size(1) - window_size + 1)]
    input_tensor = torch.stack(input_tensor)
    return input_tensor


def vector_transfer_targetFinal(batch_size, window_size, inputs_file, targets_file):
    # inputs_file shape=(1026,12,1)
    # targets_file shape=((1026,12,1))
    if inputs_file.size(0) == batch_size + (window_size - 1):
        inputs_file = inputs_file.permute(2, 0, 1)  # 交换顺序后，shape=(1,1026,12)
        input_tensor = [inputs_file[:, i:i + window_size, :].squeeze(0) for i in  # squeeze(0)后，shape=(3,12)
                        range(inputs_file.size(1) - window_size + 1)]
        input_tensor = torch.stack(input_tensor)

        # targets_file = targets_file.permute(2, 0, 1)  # 交换顺序后，shape=(1,1026,12)
        # target_tensor = [targets_file[:, i:i + window_size, :].squeeze(0) for i in  # squeeze(0)后，shape=(3,12)
        #                 range(targets_file.size(1) - window_size + 1)]
        # label_tensor = torch.stack(target_tensor)
        left_index = int((window_size - 1))
        # right_index = int(targets_file.size(0) - (window_size - 1) / 2)
        label_tensor = targets_file[left_index:, :, :].squeeze(-1)
        return input_tensor, label_tensor  # shapes are (1024,21,12) and (1024,12) respectively.
    else:
        return np.array([[0, 0]]), np.array([[0, 0]])


def vector_transfer_targetFinal_noBatch(win_size, inputs_file, targets_file):
    inputs_file = inputs_file.permute(2, 0, 1)
    input_tensor = [inputs_file[:, i:i + win_size, :].squeeze(0) for i in
                    range(inputs_file.size(1) - win_size + 1)]
    input_tensor = torch.stack(input_tensor)
    label_tensor = targets_file[int((win_size - 1)):, :, :].squeeze(-1)
    return input_tensor, label_tensor


def vector_transfer_targetFinal_noBatch2(win_size, inputs_file, targets_file):
    inputs_file = inputs_file.permute(2, 0, 1)
    input_tensor = [inputs_file[:, i:i + win_size, :].squeeze(0) for i in
                    range(inputs_file.size(1) - win_size + 1)]
    input_tensor = torch.stack(input_tensor)
    final_ind = len(targets_file)-int((win_size - 1)/2)
    label_tensor = targets_file[int((win_size - 1)/2):final_ind, :, :].squeeze(-1)
    return input_tensor, label_tensor


def vector_transfer_test_targetFinal(window_size, inputs_file):
    inputs_file = inputs_file.permute(2, 0, 1)
    input_tensor = [inputs_file[:, i:i + window_size, :].squeeze(0) for i in
                    range(inputs_file.size(1) - window_size + 1)]
    input_tensor = torch.stack(input_tensor)
    return input_tensor


def vector_transfer_test_targetFinal_new(window_size, targets_file, inputs_file):
    inputs_file = inputs_file.permute(2, 0, 1)  # 交换顺序后，shape=(1,25477,3)
    input_tensor = [inputs_file[:, i:i + window_size, :].squeeze(0) for i in  # squeeze(0)后，shape=(121,3)
                    range(0, inputs_file.size(1) - window_size + 1, window_size)]
    input_tensor = torch.stack(input_tensor)

    targets_file = targets_file.permute(2, 0, 1)  # 交换顺序后，shape=(1,25477,3)
    target_tensor = [targets_file[:, i:i + window_size, :].squeeze(0) for i in  # squeeze(0)后，shape=(121,3)
                     range(0, targets_file.size(1) - window_size + 1, window_size)]
    label_tensor = torch.stack(target_tensor)
    return input_tensor, label_tensor


def norm_one_zero(input_vector):
    minV = torch.min(input_vector)
    maxV = torch.max(input_vector)
    res = (input_vector - minV) / (maxV - minV)
    return res


class Logger(object):
    def __init__(self, filename=save_pth_path + 'default.log', stream=sys.stdout):
        self.terminal = stream
        self.log = open(filename, 'a')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.flush()  # Ensure immediate write

    def flush(self):
        self.terminal.flush()
        self.log.flush()
        # pass


# 固定随机数种子，使得模型可复现
def seed_torch(seed=1024):
    random.seed(seed)  # Python的随机性
    os.environ['PYTHONHASHSEED'] = str(seed)  # 设置Python哈希种子，为了禁止hash随机化，使得实验可复现
    np.random.seed(seed)  # numpy的随机性
    torch.manual_seed(seed)  # torch的CPU随机性，为CPU设置随机种子
    torch.cuda.manual_seed(seed)  # torch的GPU随机性，为当前GPU设置随机种子
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.   torch的GPU随机性，为所有GPU设置随机种子
    torch.backends.cudnn.benchmark = False  # if benchmark=True, deterministic will be False
    torch.backends.cudnn.deterministic = True  # 选择确定性算法


def normalization_data(train_data):
    # 训练数据归一化
    if isinstance(train_data, list):
        concatenated_data = torch.cat(train_data, dim=0)
    else:
        concatenated_data = train_data
    mean = torch.mean(concatenated_data, dim=0)
    std_dev = torch.std(concatenated_data, dim=0)
    normalized_train_data = [(sample - mean) / std_dev for sample in train_data]
    normalized_train_data = torch.stack(normalized_train_data)
    return normalized_train_data


def findLastPth(path):
    try:
        pth_files = [os.path.join(path, f) for f in os.listdir(path) if f.endswith('.pth')]
        lastPth = max(pth_files, key=os.path.getctime)
    except ValueError:
        lastPth = None
    return lastPth
