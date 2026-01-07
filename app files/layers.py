from activations import *
import numpy as np

class Layer:
    """
    Базовый класс слоя сети, который используется в качестве строительного 
    блока для построения производных классов слоев нейронной сети
    """
    def __init__(self, input_dimension, output_dimension):
        """
        Конструктор класса.

        Входные параметры:
            input_dimension (int):    Количество входных сигналов слоя
            output_dimension (int):   Количество выходных сигналов слоя
        """
        # Инициализация веса и смещения случайным равномерным шумом
        self.weights = np.random.uniform(-1, 1, (output_dimension, input_dimension))
        self.bias = np.random.uniform(-1, 1, (output_dimension, 1))
        
        # Объявление параметров слоя
        self.log = {}
        self.gradients = {}
        self.name = None
        self.function = None

    def forward(self, x):
        """
        Прямое распространение по слою

        Входные параметры:
            x (nd.array):   Активация предыдущего слоя
        Возвращаемое значение:
            (nd.array):     Активация следующего слоя
        """
        self.log["x_prev"] = x
        # Вычислить суммарный ввод для следующего слоя
        self.log["a_next"] = self.weights @ x + self.bias
        
        # Применить функцию активации
        self.log["x_next"] = self.function(self.log["a_next"])
        return self.log["x_next"]

    def backward(self, error):
        """
        Обратное распространение по слою

        Входные параметры:
            error (nd.array):   Ошибка, переданная со следующего слоя
                                 (последний слой определяется как y_hat - t)
                                 (другие скрытые слои: W.T @ error)
        Возвращаемое значение:
            (nd.array):         Ошибка, передаваемая предыдушему слою
        """
        # Использовать ошибку для вычисления дельты
        self.delta = error * self.function.derivative(self.log["x_next"])
        
        # Рассчитать и сохранить градиент для последующего обновления веса
        self.gradients["W"] = (self.delta @ self.log["x_prev"].T)
        self.gradients["b"] = self.delta
        
        # Вычислить ошибку предыдущего слоя
        error = self.weights.T @ self.delta
        
        return error

    def weights(self):
        """
        Возвращает текущую матрицу весов и вектора смещений

        Входные параметры:
            self.weights (nd.array):        Матрица весов
            self.bias (nd.array):           Вектор смещений
        """
        return self.weights, self.bias

    def _update_weights(self, lr):
        """
        Используя сохраненный градиент, эта функция обновляет внутренние 
        параметры в соответствии с алгоритмом градиентного спуска

        Входные параметры:
            lr (float):     Скорость обучения сети
        """
        self.weights -= lr * self.gradients["W"]
        self.bias -= lr * self.gradients["b"]

    def __repr__(self):
        return "[" + self.name + f" c размерностью {self.weights.shape}]"


class LinearLayer(Layer):
    """
    Слой с линейной функцией активации
    """
    def __init__(self, input_dimension, output_dimension):
        super().__init__(input_dimension, output_dimension)
        self.function = LinearFunction()
        self.name = "Линейная"

class ReLULayer(Layer):
    """
    Слой с функцией активации ReLU
    """
    def __init__(self, input_dimension, output_dimension):
        super().__init__(input_dimension, output_dimension)
        self.function = ReLU()
        self.name = "ReLU"

class TanhLayer(Layer):
    """
    Слой с функцией активации гиперболического тангенса
    """
    def __init__(self, input_dimension, output_dimension):
        super().__init__(input_dimension, output_dimension)
        self.function = Tanh()
        self.name = "Гиперболический тангенс"

class SigmoidLayer(Layer):
    """
    Слой с сигмоидальной функцией активации
    """
    def __init__(self, input_dimension, output_dimension):
        super().__init__(input_dimension, output_dimension)
        self.function = Sigmoid()
        self.name = "Сигмоида"

class LeakyReLULayer(Layer):
    """
    Слой с функцией активации ReLU с утечкой
    """
    def __init__(self, input_dimension, output_dimension):
        super().__init__(input_dimension, output_dimension)
        self.function = LeakyReLU()
        self.name = "ReLU с утечкой"

class SwishLayer(Layer):
    """
    Слой с функцией активации Swish
    """
    def __init__(self, input_dimension, output_dimension):
        super().__init__(input_dimension, output_dimension)
        self.function = Swish()
        self.name = "Swish-функция"
