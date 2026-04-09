import torch 
# Задание 1

# 1. Создаем тензор x целочисленного типа со случайным значением
x = torch.randint(1, 10, (1,), dtype=torch.int32)
print("Случайный x (int32):", x)

# 2. Преобразуем к float32 и включаем отслеживание градиента
x = x.to(dtype=torch.float32)
x.requires_grad = True
print("x после преобразования:", x)

# 3. Выполняем операции
n = 2

k = torch.randint(1, 11, (1,), dtype=torch.float32)
print("Случайный множитель k:", k)

# вычисления
y = torch.exp((x ** n) * k)

print("Результат после операций:", y)

# 4. Вычисляем производную dy/dx
y.backward()

print("Производная dy/dx:", x.grad)