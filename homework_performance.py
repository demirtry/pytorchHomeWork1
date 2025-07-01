import time
import torch


def cpu_timer(func, *args, **kwargs) -> float:
    '''
    Таймер для замера времени вычислений на CPU
    '''
    start_time = time.time()
    func(*args, **kwargs)
    end_time = time.time()
    execution_time = round((end_time - start_time) * 10**3, 2)

    return execution_time


def gpu_timer(func, *args, **kwargs) -> float:
    """
    Таймер для замера времени вычислений на GPU
    """
    torch.cuda.synchronize()
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    start_event.record()
    func(*args, **kwargs)
    end_event.record()

    torch.cuda.synchronize()
    execution_time = round(start_event.elapsed_time(end_event), 2)

    return execution_time


def matrix_mul(matrix: torch.Tensor):
    """
    Функция для вычисления матричного умножения
    """
    return torch.matmul(matrix, matrix.mT)


def elementwise_sum(matrix: torch.Tensor):
    """
    Функция для вычисления поэлементного сложения
    """
    return matrix + matrix


def elementwise_mul(matrix: torch.Tensor):
    """
    Функция для вычисления поэлементного умножения
    """
    return matrix * matrix


def transpose(matrix: torch.Tensor):
    """
    Функция для транспонирования матрицы
    """
    return matrix.mT


def sum_all_elements(matrix: torch.Tensor):
    """
    Функция для вычисления суммы всех элементов
    """
    return torch.sum(matrix)


def plot_table(cpu_times: list[list[float]], gpu_times: list[list[float]]) -> None:
    """
    Функция для отображения результатов в таблице
    :param cpu_times: время на CPU для каждой функции
    :param gpu_times: время на GPU для каждой функции
    :return: None
    """
    operations = [
        "Матричное умножение",
        "Поэлементное сложение",
        "Поэлементное умножение",
        "Транспонирование",
        "Сумма всех элементов",
    ]
    for idx in range(len(matrix_lst)):
        print(f"\nРезультаты для Матрицы {idx+1}:\n")
        print(f"{'Операция':<25} | {'CPU (мс)':>7} | {'GPU (мс)':>7} | {'Ускорение':>9}")
        print("-"*65)
        for i, op in enumerate(operations):
            cpu_t = cpu_times[idx][i]
            gpu_t = gpu_times[idx][i]
            try:
                speedup = round(cpu_t / gpu_t, 1) if gpu_t != 0 else float('inf')
            except ZeroDivisionError:
                speedup = float('inf')
            print(f"{op:<25} | {cpu_t:>7.2f} | {gpu_t:>7.2f} | {speedup:>8.1f}x")


if __name__ == "__main__":

    # создаю матрицы нужных размеров
    matrix_1 = torch.randn(64, 1024, 1024)
    matrix_2 = torch.randn(128, 512, 512)
    matrix_3 = torch.randn(256, 256, 256)

    matrix_lst = [matrix_1, matrix_2, matrix_3]

    print("CUDA доступно:", torch.cuda.is_available())

    # Копии матриц на GPU (если доступно)
    matrix_lst_gpu = []
    if torch.cuda.is_available():
        device = torch.device('cuda')
        for matrix in matrix_lst:
            matrix_lst_gpu.append(matrix.to(device))
    else:
        matrix_lst_gpu = [None] * len(matrix_lst)

    cpu_times = []
    gpu_times = []

    for i in range(len(matrix_lst)):
        cpu_current = []
        gpu_current = []

        # вычисления на CPU
        cpu_current.append(cpu_timer(matrix_mul, matrix_lst[i]))
        cpu_current.append(cpu_timer(elementwise_sum, matrix_lst[i]))
        cpu_current.append(cpu_timer(elementwise_mul, matrix_lst[i]))
        cpu_current.append(cpu_timer(transpose, matrix_lst[i]))
        cpu_current.append(cpu_timer(sum_all_elements, matrix_lst[i]))

        # вычисления на GPU
        if torch.cuda.is_available():
            gpu_current.append(gpu_timer(matrix_mul, matrix_lst_gpu[i]))
            gpu_current.append(gpu_timer(elementwise_sum, matrix_lst_gpu[i]))
            gpu_current.append(gpu_timer(elementwise_mul, matrix_lst_gpu[i]))
            gpu_current.append(gpu_timer(transpose, matrix_lst_gpu[i]))
            gpu_current.append(gpu_timer(sum_all_elements, matrix_lst_gpu[i]))
        else:
            gpu_current = [0.0] * 5  # если GPU недоступно заполню нулями

        cpu_times.append(cpu_current)
        gpu_times.append(gpu_current)

    plot_table(cpu_times, gpu_times) # построение таблицы


"""
Анализ результатов:
1) Какие операции получают наибольшее ускорение на GPU?
Для матричного умножения, поэлементного сложения и поэлементного умножения на GPU получено наибольшее ускорение.
Для всех трех матриц операция транспонирования на GPU не получила ускорение.

2) Почему некоторые операции могут быть медленнее на GPU?
Если объем данных маленький, то время копирования с CPU на GPU может превышать выигрыш от скорости вычислений.

3)Как размер матриц влияет на ускорение?
Чем больше матрица, тем выше потенциальное ускорение на GPU. 

4) Что происходит при передаче данных между CPU и GPU?
Сначала происходит выделение памяти на GPU (VRAM), затем данные копируются с CPU (RAM) и форматируются 
"""