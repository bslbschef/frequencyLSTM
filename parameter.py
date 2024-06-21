import glob
import os

import torch.nn as nn
# 公共参数设置
input_dim = 1  # 输入特征维度
hidden_dim = 3  # LSTM隐藏层维度
output_dim = 1  # 输出维度
num_layers = 2  # LSTM层数
window_size = 81  # 设置一个序列长度(奇数)
flag_bidirectional = True
if flag_bidirectional:
    time_flag = 2
else:
    time_flag = 1

learning_rate = 0.001  # 学习率
lr_update_coefficient = 0.1  # 学习率更新的缩小系数
lr_update_patient = 20  # 学习率更触发更新的容忍度
criterion1 = nn.MSELoss()  # 使用均方误差Loss or 使用L1Loss
criterion2 = nn.L1Loss()  # 使用均方误差Loss or 使用L1Loss
criterion3 = nn.MSELoss()  # 使用均方误差Loss or 使用L1Loss
# 定义每个损失的权重
weight1 = 1
weight2 = 1
weight3 = 1

num_epochs = 48000  # 训练轮次
# 数据地址
train_path = "./data/train/"
train_csv_files = glob.glob(os.path.join(train_path, '*.csv'))
test_path = "./data/test/"
test_csv_files = glob.glob(os.path.join(train_path, '*.csv'))

# 模型保存路径
save_pth_path = './result/v1/'
