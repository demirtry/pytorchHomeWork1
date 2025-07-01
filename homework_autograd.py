import torch

# Задание 2: Автоматическое дифференцирование

# 2.1 Простые вычисления с градиентами
# Создам тензоры x, y, z с requires_grad=True
x = torch.tensor(2.0, requires_grad=True)
y = torch.tensor(3.0, requires_grad=True)
z = torch.tensor(5.0, requires_grad=True)
print(f"Тензоры x, y, z:\n{x} {y} {z}")

# Вычисляю функцию f(x,y,z) = x^2 + y^2 + z^2 + 2*x*y*z
f = x**2 + y**2 + z**2 + 2*x*y*z
print(f"Значение функции: {f}")

# Вычисление градиентов
f.backward()

grad_x = x.grad
grad_y = y.grad
grad_z = z.grad

print("Вычисленные градиенты:")
print(f"grad_x = {grad_x}")
print(f"grad_y = {grad_y}")
print(f"grad_z = {grad_z}\n")

# Проверка
# Найду частные производные:
# df/dx = 2x + 2yz
# df/dy = 2y + 2xz
# df/dz = 2z + 2xy

review_grad_x = 2*x + 2*y*z
review_grad_y = 2*y + 2*x*z
review_grad_z = 2*z + 2*x*y

assert review_grad_x == grad_x, f"Ошибка в grad_x: {review_grad_x} != {grad_x}"
assert review_grad_y == grad_y, f"Ошибка в grad_y: {review_grad_y} != {grad_y}"
assert review_grad_z == grad_z, f"Ошибка в grad_z: {review_grad_z} != {grad_z}"


# Реализация функции MSE (Mean Squared Error):

def mse_loss(y_true: torch.Tensor, x: torch.Tensor, w: torch.Tensor, b: torch.Tensor):
    """
    Вычисляет MSE и градиенты по w и b

    :argument
    x - входные данные (тензор)
    y_true - истинные значения (тензор)
    w - вес (тензор с requires_grad=True)
    b - смещение (тензор с requires_grad=True)

    :return
    loss - значение MSE
    grads - словарь с градиентами {'w_grad': ..., 'b_grad': ...}
    """
    # x и y_true должны иметь один размер
    if x.shape != y_true.shape:
        raise ValueError("x.shape != y_true.shape")

    # PyTorch должен отслеживать градиенты
    if not w.requires_grad:
        raise ValueError("w must contain requires_grad == True")
    if not b.requires_grad:
        raise ValueError("b must contain requires_grad == True")

    # Вычисление y_pred
    y_pred = w * x + b

    # Вычисление MSE
    loss = torch.sum((y_pred - y_true) ** 2) / len(x)

    # Вычисление градиентов
    loss.backward()

    # Создаю словарь для удобства возвращения
    grads = {
        'w_grad': w.grad.item(),
        'b_grad': b.grad.item()
    }

    return loss, grads


def composite_func(x: torch.Tensor):
    """
    Вычисляет градиент df/dx используя sin(x^2 + 1)
    """
    assert x.requires_grad == True, "x must contain requires_grad == True"

    def func(x):
        """:return sin(x^2 + 1)"""
        return torch.sin(x**2 + 1)

    y = func(x)
    y.backward()
    return x.grad.item()


if __name__ == "__main__":
    # Пример использования функции MSE

    # Инициализация входных параметров
    x = torch.tensor([1.0, 2.0, 3.0, 4.0])
    y_true = torch.tensor([2.0, 4.0, 6.0, 8.0])
    w = torch.tensor(1.0, requires_grad=True)
    b = torch.tensor(0.0, requires_grad=True)

    loss, grads = mse_loss(x, y_true, w, b)

    print(f"MSE: {loss:.4f}")
    print(f"Градиент по w: {grads['w_grad']}")
    print(f"Градиент по b: {grads['b_grad']}")

    # Использование составной функции и проверка с помощью torch.autograd.grad
    x = torch.tensor(2.0, requires_grad=True)
    result = composite_func(x)
    print(f"результат составной функции: {result}")
    review_result = torch.autograd.grad(torch.sin(x**2 + 1), x)[0].item()
    print(f"Проверка через autograd.grad: {review_result}")

    assert result == review_result, "Результат отличается"
