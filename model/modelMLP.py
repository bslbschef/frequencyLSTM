import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# 定义超参数
input_size = 1
output_size = 1
hidden_size = 50
sequence_length = 100  # 设定一个固定的序列长度
batch_size = 10
num_epochs = 5
learning_rate = 0.001

# 数据生成（示例数据，实际使用时应根据具体任务生成）
num_samples = 100
max_sequence_length = 300  # 最大序列长度
data = [np.random.rand(np.random.randint(1, max_sequence_length), input_size).astype(np.float32) for _ in range(num_samples)]
labels = [np.random.rand(len(seq), output_size).astype(np.float32) for seq in data]

# 数据填充到固定长度
def pad_sequences(sequences, maxlen):
    padded_sequences = np.zeros((len(sequences), maxlen, input_size))
    for i, seq in enumerate(sequences):
        padded_sequences[i, :len(seq)] = seq
    return padded_sequences

data_padded = pad_sequences(data, sequence_length)
labels_padded = pad_sequences(labels, sequence_length)

# 转换为PyTorch张量
data_padded = torch.tensor(data_padded)
labels_padded = torch.tensor(labels_padded)

# 自定义Dataset类
class SequenceDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

dataset = SequenceDataset(data_padded, labels_padded)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# 定义MLP模型
class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        batch_size, seq_len, _ = x.size()
        x = x.view(batch_size * seq_len, -1)
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = out.view(batch_size, seq_len, -1)
        return out

model = MLP(input_size, hidden_size, output_size)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# 训练模型
model.train()
for epoch in range(num_epochs):
    for i, (inputs, targets) in enumerate(dataloader):
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (epoch+1) % 1 == 0 and (i+1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(dataloader)}], Loss: {loss.item():.4f}')

# 模型评估（简单的评估）
model.eval()
with torch.no_grad():
    inputs, targets = next(iter(dataloader))
    outputs = model(inputs)
    loss = criterion(outputs, targets)
    print(f'Evaluation Loss: {loss.item():.4f}')
