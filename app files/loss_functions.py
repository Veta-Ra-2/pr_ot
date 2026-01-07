import numpy as np

"""
Базовый класс для наследования
"""
class BaseLoss():
    def __init__(self):
        self.log = {}
        self.name = None
    def __call__(self, x, y):
        pass
    def backward(self):
        pass
    def __repr__(self):
        return self.name

"""
Среднеквадратическая функция ошибки
"""
class MSELoss(BaseLoss):
    """
    Обеспечивает вызов экземпляра класса как функции
    """
    def __call__(self, t, y_hat):
        self.log["t"] = t
        self.log["y_hat"] = y_hat
        return np.mean(0.5 * (t - y_hat)**2)

    def backward(self):
        """
        Производная по вектору весов.
        Возвращаемое значение - массив y_hat - t
        """
        return (self.log["y_hat"] - self.log["t"])
