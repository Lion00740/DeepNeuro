# Попробуем обучить один нейрон на задачу классификации двух классов

import pandas as pd # библиотека pandas нужна для работы с данными
import matplotlib.pyplot as plt # matplotlib для построения графиков
import numpy as np # numpy для работы с векторами и матрицами
import torch
import torch.nn as nn

df = pd.read_csv('data.csv')

# выделим целевую переменную в отдельную переменную
y = df.iloc[:, 4].values

# так как ответы у нас строки - нужно перейти к численным значениям
y = np.where(y == "Iris-setosa", 1, -1)

# возьмем два признака
X = df.iloc[:, [0, 2]].values

# Признаки в X, ответы в y
plt.figure()
plt.scatter(X[y==1, 0], X[y==1, 1], color='red', marker='o')
plt.scatter(X[y==-1, 0], X[y==-1, 1], color='blue', marker='x')

# Преобразуем данные в тензоры PyTorch
X_tensor = torch.FloatTensor(X)
y_tensor = torch.FloatTensor(y).reshape(-1, 1)  # Изменяем форму для совместимости

# Создаем линейную модель (нейрон)
# 2 входных признака, 1 выход (бинарная классификация)
linear_model = nn.Linear(2, 1)

print('Начальные веса модели:')
print('w (веса):', linear_model.weight.data)
print('b (смещение):', linear_model.bias.data)

loss_fn = nn.MSELoss()  # Среднеквадратичная ошибка

# Оптимизатор SGD (стохастический градиентный спуск)
optimizer = torch.optim.SGD(linear_model.parameters(), lr=0.01)

# Обучение
num_epochs = 1000  # Количество эпох обучения

for i in range(num_epochs):
    # Прямой проход: получаем предсказания
    predictions = linear_model(X_tensor)
    
    # Вычисляем ошибку
    loss = loss_fn(predictions, y_tensor)
    
    # Обратное распространение ошибки
    optimizer.zero_grad()  # Обнуляем градиенты
    loss.backward()        # Вычисляем градиенты
    
    # Обновляем веса
    optimizer.step()
    
    print('Ошибка на ' + str(i+1) + ' итерации: ', loss.item())