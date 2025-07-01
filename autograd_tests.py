import torch
import unittest
from homework_autograd import mse_loss, composite_func


class TestAutogradFunctions(unittest.TestCase):
    def setUp(self):
        self.x = torch.tensor(2.0, requires_grad=True)
        self.y = torch.tensor(3.0, requires_grad=True)
        self.z = torch.tensor(5.0, requires_grad=True)

    def test_gradients(self):
        f = self.x ** 2 + self.y ** 2 + self.z ** 2 + 2 * self.x * self.y * self.z
        f.backward()

        expected_grad_x = 2 * self.x + 2 * self.y * self.z
        expected_grad_y = 2 * self.y + 2 * self.x * self.z
        expected_grad_z = 2 * self.z + 2 * self.x * self.y

        self.assertEqual(self.x.grad, expected_grad_x)
        self.assertEqual(self.y.grad, expected_grad_y)
        self.assertEqual(self.z.grad, expected_grad_z)

    def test_mse_loss(self):
        x = torch.tensor([1.0, 2.0, 3.0, 4.0])
        y_true = torch.tensor([2.0, 4.0, 6.0, 8.0])
        w = torch.tensor(1.0, requires_grad=True)
        b = torch.tensor(0.0, requires_grad=True)

        loss, grads = mse_loss(y_true, x, w, b)

        # Проверяю loss
        expected_loss = torch.sum((x - y_true) ** 2) / len(x)
        self.assertAlmostEqual(loss.item(), expected_loss.item(), places=4)

        # Проверяю градиенты
        expected_w_grad = torch.sum(2 * (w * x + b - y_true) * x) / len(x)
        expected_b_grad = torch.sum(2 * (w * x + b - y_true)) / len(x)

        self.assertAlmostEqual(grads['w_grad'], expected_w_grad.item(), places=4)
        self.assertAlmostEqual(grads['b_grad'], expected_b_grad.item(), places=4)

    def test_composite_func(self):
        x = torch.tensor(2.0, requires_grad=True)
        result = composite_func(x)

        # Проверяем с помощью autograd.grad
        expected_result = torch.autograd.grad(torch.sin(x ** 2 + 1), x)[0].item()
        self.assertAlmostEqual(result, expected_result, places=4)

    def test_mse_input_validation(self):
        with self.assertRaises(ValueError):
            x = torch.tensor([1.0, 2.0])
            y_true = torch.tensor([1.0])
            w = torch.tensor(1.0, requires_grad=True)
            b = torch.tensor(0.0, requires_grad=True)
            mse_loss(y_true, x, w, b)

        with self.assertRaises(ValueError):
            x = torch.tensor([1.0, 2.0])
            y_true = torch.tensor([1.0, 2.0])
            w = torch.tensor(1.0, requires_grad=False)
            b = torch.tensor(0.0, requires_grad=True)
            mse_loss(y_true, x, w, b)