import torch
import torch.nn as nn
import pandas as pd
import numpy as np

# без пасхалки....

# Загружаем датасет
df = pd.read_csv('dataset_simple.csv')

print(df.head())

# Признаки
X = torch.Tensor(df.iloc[:, [0,1]].values)

# Целевая переменная
y = df.iloc[:, 2].values
y = torch.Tensor(np.where(y == 1, 1, -1).reshape(-1,1))


# Создаем нейросеть
class NNet(nn.Module):
    def __init__(self, in_size, hidden_size, out_size):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(in_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, out_size),
            nn.Tanh()
        )

    def forward(self, X):
        return self.layers(X)


# параметры сети
inputSize = 2     # возраст + доход
hiddenSize = 5
outputSize = 1

net = NNet(inputSize, hiddenSize, outputSize)

# функция ошибки и оптимизатор
lossFn = nn.MSELoss()
optimizer = torch.optim.SGD(net.parameters(), lr=0.01)


# Обучение
epochs = 400

for i in range(epochs):
    pred = net(X)
    loss = lossFn(pred, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if i % 20 == 0:
        print(f"Эпоха {i}, ошибка: {loss.item()}")


# Проверка
with torch.no_grad():
    pred = net.forward(X)

pred = torch.Tensor(np.where(pred >=0, 1, -1).reshape(-1,1))
err = sum(abs(y-pred))/2

print("\nКоличество ошибок:", err.item())