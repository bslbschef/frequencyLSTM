import glob
import os

import torch.nn as nn
# 公共参数设置
from loss.stdLoss import StdDevLoss

input_dim = 1  # 输入特征维度
hidden_dim = 64  # LSTM隐藏层维度
output_dim = 1  # 输出维度
num_layers = 2  # LSTM层数
flag_bidirectional = True
if flag_bidirectional:
    time_flag = 2
else:
    time_flag = 1

learning_rate = 0.001  # 学习率
lr_update_coefficient = 0.1  # 学习率更新的缩小系数
lr_update_patient = 20  # 学习率更触发更新的容忍度
# criterion = nn.L1Loss()  # 使用均方误差MSELoss or 使用L1Loss
criterion = StdDevLoss(alpha=0.1)


num_epochs = 10000  # 训练轮次
# 数据地址
train_path = "./data/train/"
train_csv_files = glob.glob(os.path.join(train_path, '*.csv'))
test_path = "./data/test/"
test_csv_files = glob.glob(os.path.join(train_path, '*.csv'))

# 模型保存路径
save_pth_path = './result/v1/'
