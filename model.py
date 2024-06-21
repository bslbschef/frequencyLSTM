import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# 定义超参数
input_size = 1
hidden_size = 50
num_layers = 1
output_size = 1
sequence_length = 1000
num_epochs = 5
learning_rate = 0.001

# 生成随机数据
num_samples = 10
data = np.random.rand(num_samples, sequence_length, input_size).astype(np.float32)
labels = np.random.randint(2, size=(num_samples, output_size)).astype(np.float32)

# 转换为PyTorch张量
data = torch.tensor(data)
labels = torch.tensor(labels)

# 定义LSTM模型
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        out, _ = self.lstm(x, (h0, c0))
        out = out[:, -1, :]
        out = self.fc(out)
        return out

# 实例化模型、定义损失函数和优化器
model = LSTMModel(input_size, hidden_size, num_layers, output_size)
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# 训练模型
model.train()
for epoch in range(num_epochs):
    outputs = model(data)
    loss = criterion(outputs, labels)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if (epoch+1) % 1 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# 模型评估（简单的准确率计算）
model.eval()
with torch.no_grad():
    outputs = model(data)
    predicted = (torch.sigmoid(outputs) > 0.5).float()
    accuracy = (predicted == labels).float().mean()
    print(f'Accuracy: {accuracy:.4f}')
