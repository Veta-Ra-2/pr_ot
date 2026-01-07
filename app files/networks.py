from layers import LinearLayer, ReLULayer, TanhLayer, SigmoidLayer, LeakyReLULayer, SwishLayer
import json
from pathlib import Path
import pickle
from auxiliary import save_pickle, load_pickle

class NeuralNetwork():
    """
    Базовый класс нейронной сети, который объединяет несколько слоев. 
    Использует их, чтобы реализовать прямое и обратное распространение.
    """
    def __init__(self, layers = None, load_s_weights = False, model_fp = None):
        """
        Конструктор класса.
        Входные параметры:
            layers (list):              Список слоев сети
            load_s_weights (bool):      Если установлен этот параметр, вместо обучения загрузить предобученную модель
            model_fp (str):             Путь к сохраненным предварительно обученным весам
        """
        if not load_s_weights and layers:
            # если веса не должны быть загружены и на вход передан список слоев, использовать их
            self.layers = layers
        elif load_s_weights and model_fp:
            # иначе (веса должны быть загружены и путь указан) загрузить веса
            self.layers = self._load_layers(model_fp)
        else:
            # иначе (путь для загрузки весов не указан) сообщить об ошибке
            raise ValueError("Не указан путь для загрузки весов.")

    def forward(self, x):
        """
        Выполняет прямое распространение по нейронной сети
        Входные параметры:
            x (nd.array):               Вход нейронной сети
        Возвращаемое значение:
            x (nd.array):               Выход нейронной сети
        """
        # Осуществить прямое распространение по каждому слою сети и вернуть результат
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def _backpropagation(self, error):
        """
        Распространяет ошибку по сети в обратном направлении.
        Во время обратного распространения ошибка сохраняется 
        каждым внутренним слоем для последующего обновления

        Входные параметры:
            error (nd.array):           Величина ошибки функции потерь
        """
        # Осуществить обратное распространение по каждому слою сети (в обратном порядке)
        for layer in reversed(self.layers):
            # с вычислением ошибки 
            error = layer.backward(error)

    def _update_weights(self, lr):
        """
        Обновление весов с использованием ранее сохраненных градиентов в самих слоях

        Входные параметры:
            lr (float):                 Скорость обучения сети
        """
        # Обновить весовые коэффициенты на каждом слое сети 
        for layer in self.layers:
            layer._update_weights(lr)

    def _save_model(self, model_path):
        """
        Сохраняет структуру сети в pickle-файл в виде словаря

        Входные параметры:
            model_path (str):           Путь к директории для сохранения модели (заканчивается символом /)
        """
        # Создать словарь для сохранения информации о модели
        dic = {}

        # Собрать всю информацию о модели (по всем слоям) и записать в словарь
        for i, layer in enumerate(self.layers):
            dic[f"layer_{i+1}"] = {}
            dic[f"layer_{i+1}"]["type"] = layer.name
            dic[f"layer_{i+1}"]["weight_shape"] = layer.weights.shape
            dic[f"layer_{i+1}"]["weights"] = layer.weights
            dic[f"layer_{i+1}"]["bias"] = layer.bias

        # Если директория по указанному пути не создана, создать ее
        Path(model_path).mkdir(exist_ok=True)

        # Сохранить словарь в pickle-файл
        save_pickle(dic, model_path + "model.pickle")


    def _load_layers(self, model_fp):
        """
        Функция для загрузки сохраненной архитектуры модели и 
        весов и инициализации с этими данными новой нейронной сети.

        Входные параметры:
            model_fp (str):             Путь к сохраненному pickle-файлу

        Возвращаемое значение:
            layers (list):              Список слоев сети
        """
        # Словарь с названиями функций активации и соответствующими им слоями
        layer_dict = {"Линейная": LinearLayer,
                        "ReLU": ReLULayer,
                        "Гиперболический тангенс": TanhLayer,
                        "Сигмоида": SigmoidLayer,
                        "ReLU с утечкой": LeakyReLULayer,
                        "Swish-функция": SwishLayer}

        # Загрузить сохраненные параметры модели в словарь
        dic = load_pickle(model_fp)

        # Список для хранения созданных (загружаемых из файла) слоев
        layers = []

        # Для каждого сохраненного слоя выполнить:
        for i in range(len(dic)):
            # Получить информацию из словаря
            layer_type = dic[f"layer_{i+1}"]["type"]
            layer_w_sh = dic[f"layer_{i+1}"]["weight_shape"]
            layer_weights = dic[f"layer_{i+1}"]["weights"]
            layer_bias = dic[f"layer_{i+1}"]["bias"]

            # Создать новый слой того же типа и заменить его веса сохраненными
            new_layer = layer_dict[layer_type](layer_w_sh[1], layer_w_sh[0])
            new_layer.weights = layer_weights
            new_layer.bias = layer_bias

            # Добавить новый слой в список слоев
            layers.append(new_layer)

        return layers