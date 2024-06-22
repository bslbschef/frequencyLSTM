import numpy as np
import torch.optim as optim
import sys
from torch.optim.lr_scheduler import ReduceLROnPlateau
from datetime import datetime
from model import *
from util import *
from parameter import *
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.metrics import r2_score, mean_squared_error
from math import sqrt
import statistics
from array import array
from pytorch_msssim import ssim, ms_ssim, SSIM, MS_SSIM


# 固定随机数种子，使得模型可复现
seed_torch()
# 设置模型参数
sys.stdout = Logger(stream=sys.stdout)
begin_time = datetime.now()
model = WindSpeedGRU(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim, num_layers=num_layers, flag_bidirectional=flag_bidirectional, time_flag=time_flag)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=lr_update_coefficient, patience=lr_update_patient, verbose=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
# 加载最新模型
lastPatFile = findLastPth(save_pth_path)
if lastPatFile is not None:
    checkpoint = torch.load(lastPatFile)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch'] + 1  # 从上次训练的下一个epoch开始
    loss = checkpoint['loss']
    learning_rate = 0.0001  # checkpoint['learning_rate']*1000 [4820kaishi]
    cost_time = checkpoint['cost_time']
    for param_group in optimizer.param_groups:
        param_group['lr'] = learning_rate
    print('----------------------------------------------------------------------------------')
    print("Loading last model successful !\n"+"Last Epoch: " + str(start_epoch-1))
else:
    start_epoch = 1
    cost_time = datetime.now() - datetime.now()

# 加载训练数据和标签
train_data = []
for i in train_csv_files:
    wind_data = load_data(i).to(device)
    train_data.append(wind_data)

# 训练数据归一化(已经在matlab处理了)
# train_data = normalization_data(train_data)
# 训练模型
losses = []
for epoch in range(start_epoch, num_epochs+1):
    total_loss = []
    all_predictions, all_targets = [], []
    r2, rmse = [], []
    for i in range(len(train_csv_files)):
        cur_train_data = train_data[i]
        inputs, targets = cur_train_data[:, 0, :], cur_train_data[:, 1, :]

        optimizer.zero_grad()
        new_inputs = inputs
        new_targets = targets
        outputs = model(new_inputs)
        loss = criterion(outputs, new_targets)
        loss.backward()
        clip_value = 1.0
        for param in model.parameters():
            if param.grad is not None:
                param.grad.data = torch.clamp(param.grad.data, -clip_value, clip_value)
        optimizer.step()

        total_loss.append(loss.item())
        all_predictions = outputs.detach().cpu().numpy()
        all_targets = new_targets.detach().cpu().numpy()
        r2_cur = r2_score(all_targets, all_predictions)
        rmse_cur = sqrt(mean_squared_error(all_targets, all_predictions))
        r2.append(r2_cur)
        rmse.append(rmse_cur)

    epoch_loss = np.mean(total_loss)
    losses.append(epoch_loss)
    scheduler.step(loss)  # 根据loss更新lr
    current_lr = optimizer.param_groups[0]['lr']
    print('----------------------------------------------------------------------------------')
    print(f'Epoch [{epoch}/{num_epochs}], Current Learning Rate: {current_lr}')
    print(f'Epoch [{epoch}/{num_epochs}], Current Cost Time: {datetime.now() - begin_time}')
    print(f'Epoch [{epoch}/{num_epochs}], Cumulative Cost Time: {datetime.now() - begin_time + cost_time}')
    print(f'Epoch [{epoch}/{num_epochs}], Loss: {epoch_loss:.9f}')
    print(f'Epoch [{epoch}/{num_epochs}],  R2: {np.mean(r2):.4f}, RMSE: {np.mean(rmse):.4f}')
    if epoch % 10 == 0:
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            'learning_rate': optimizer.param_groups[0]['lr'],
            'cost_time': datetime.now() - begin_time + cost_time,
        }, save_pth_path + 'epoch_spectrum' + str(epoch).zfill(4) + '.pth')

# 绘制Loss变化曲线
fig, ax = plt.subplots()
plt.figure(2, figsize=(8, 5))
plt.plot(losses[5:], color='blue')
plt.xlabel('Epochs')
plt.ylabel('MSELoss')
plt.title('Training Mean MSELoss')
plt.savefig(save_pth_path + 'loss-w.png')
plt.show(block=False)

