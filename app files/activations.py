import numpy as np

class BaseFunction:
    """
    Базовый класс функции активации
    """
    def __init__(self):
        self.name = None

    def __call__(self):
        pass

    def __repr__(self):
        return self.name

    """
    Производная функции
    """
    def derivative(self):
        pass


class LinearFunction(BaseFunction):
    """
    Линейная функция активации
    """
    def __init__(self):
        self.name = "Линейная"

    def __call__(self, x):
        return x

    def derivative(self, x):
        return 1


class ReLU(BaseFunction):
    """
    Функция активации ReLU "Выпрямитель"
    (Rectified Linear Unit)
    """
    def __init__(self):
        self.name = "ReLU"

    def __call__(self, x):
        x[x < 0] = 0
        return x

    def derivative(self, x):
        x[x < 0] = 0
        x[x > 0] = 1
        return x


class Tanh(BaseFunction):
    """
    Функция активации гиперболического тангенса
    """
    def __init__(self):
        self.name = "Гиперболический тангенс"

    def __call__(self, x):
        return np.tanh(x)

    def derivative(self, x):
        x = 1 - self(x)**2
        return x


class Sigmoid(BaseFunction):
    """
    Сигмоидальная функция активации
    """
    def __init__(self):
        self.name = "Сигмоида"

    def __call__(self, x):
        return 1 / (1 + np.exp(-x))

    def derivative(self, x):
        return self(x)*(1-self(x))

class LeakyReLU(BaseFunction):
    """
    Функция активации ReLU "с утечкой"
    """
    def __init__(self):
        self.name = "ReLU с утечкой"

    def __call__(self, x, c=0.01):
        self.c = c
        x[x < 0] *= c
        return x

    def derivative(self, x):
        x[x < 0] = self.c
        x[x > 0] = 1
        return x

class Swish(BaseFunction):
    """
    Функция активации Swish
    """
    def __init__(self):
        self.name = "Swish-функция"

    def __call__(self, x, beta=1):
        self.beta = beta
        return x * self._sigmoid(beta * x)

    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def derivative(self, x):
        return self.beta * self(self.beta * x) + self._sigmoid(self.beta * x) * (1 - self.beta * self(self.beta * x))
