import torch

# Задание 1: Создание и манипуляции с тензорами

# 1.1 Создание тензоров
# - Тензор размером 3x4, заполненный случайными числами от 0 до 1
print("Тензор размером 3x4, заполненный случайными числами от 0 до 1:")
tensor1 = torch.rand(3, 4)
print(tensor1)
print()

# - Тензор размером 2x3x4, заполненный нулями
print("Тензор размером 2x3x4, заполненный нулями:")
tensor2 = torch.zeros(2, 3, 4)
print(tensor2)
print()

# - Тензор размером 5x5, заполненный единицами
print("Тензор размером 5x5, заполненный единицами:")
tensor3 = torch.ones(5, 5)
print(tensor3)
print()

# - Тензор размером 4x4 с числами от 0 до 15 (используйте reshape)
print("Тензор размером 4x4 с числами от 0 до 15:")
tensor4 = torch.arange(16).reshape(4, 4)
print(tensor4)
print()

# 1.2 Операции с тензорами
# Создам тензоры А и В нужных размеров из случайных чисел от 0 до 1
print("Тензор А размером 3x4:")
tensorA = torch.rand(3, 4)
print(tensorA)
print()

print("Тензор B размером 4x3:")
tensorB = torch.rand(4, 3)
print(tensorB)
print()

# Выполню транспонирование тензора A
print("Транспонирование тензора A:")
A_transposed = tensorA.T
print(A_transposed)
print()

# Выполню матричное умножение A и B (результат 3x3)
print("Матричное умножение A и B:")
matrix_mul = torch.matmul(tensorA, tensorB)
print(matrix_mul)
print()

# Выполню поэлементное умножение A и транспонированного B
print("Поэлементное умножение A и B.T:")
elementwise = tensorA * tensorB.T
print(elementwise)
print()

# Сумма всех элементов тензора A
print("Сумма всех элементов тензора A:")
sum_A = torch.sum(tensorA)
print(sum_A)
print()

# 1.3 Индексация и срезы
# Создам тензор 5x5x5 с числами от 0 до 124, с использованием reshape
print("Тензор 5x5x5:")
tensor = torch.arange(125).reshape(5, 5, 5)
print(tensor)
print()

# Первая строка (5,5,5) значит (глубина, строки, столбцы)
print("Первая строка (второе измерение) каждого среза:")
first_row = tensor[:, 0, :]
print(first_row)
print()

# Последний столбец (третье измерение)
print("Последний столбец (третье измерение) каждого среза:")
last_column = tensor[:, :, -1]
print(last_column)
print()

# Подматрица 2x2 из центра тензора
print("Подматрица 2x2 из центра")
center_submatrix = tensor[2, 2:4, 2:4]  # т.к. в матрице 5х5 центр 2х2 определен неоднозначно, выбрал индексы 2 и 3
print(center_submatrix)
print()

# Все элементы с четными индексами
print("Элементы с четными индексами по всем измерениям:")
even_index = tensor[::2, ::2, ::2]
print(even_index)
print()

# 1.4 Работа с формами
# Создам тензор из 24 элементов (использую arange для наглядности)
print("Тензор из 24 элементов:")
tensor_1x24 = torch.arange(24)
print(tensor_1x24)
print()

# Преобразование в 2x12
print("Форма 2x12:")
tensor_2x12 = tensor_1x24.view(2, 12)
print(tensor_2x12)
print()

# Преобразование в 3x8
print("Форма 3x8:")
tensor_3x8 = tensor_1x24.view(3, 8)
print(tensor_3x8)
print()

# Преобразование в 4x6
print("Форма 4x6:")
tensor_4x6 = tensor_1x24.view(4, 6)
print(tensor_4x6)
print()

# Преобразование в 2x3x4
print("Форма 2x3x4:")
tensor_2x3x4 = tensor_1x24.view(2, 3, 4)
print(tensor_2x3x4)
print()

# Преобразование в 2x2x2x3
print("Форма 2x2x2x3:")
tensor_2x2x2x3 = tensor_1x24.view(2, 2, 2, 3)
print(tensor_2x2x2x3)
print()