import numpy as np
import matplotlib.pyplot as plt
import pickle
from loss_functions import MSELoss

def calc_perf(X, Y, network, plot=False):
    """
    Принимает на вход образцы X и метки Y, использует сеть, 
    чтобы сделать прогноз и вычислить количество правильных прогнозов и MSE.

    Входные параметры:
        X (nd.array):               Обучающая выборка [экземпляры x признаки]
        Y (nd.array):               Метки обучающей выборки [экземпляры x 1]
        network (NeuralNetwork):    нейронная сеть
        plot (boolean):             строить график или нет
    Возвращаемые значения:
        mse_error                   Функция потерь (среднеквадратическая ошибка)
        accuracy (float):           Процент корректно классифицированных экземпляров
        prediction (nd.array):      Результаты бинарной классификации данных (0 и 1)
    """
    # Получить предсказания нейронной сети
    prediction = network.forward(X.T)[0]

    # Инициализировать среднеквадратическую ошибку (функцию потерь)
    loss_function = MSELoss()

    # Вычислить значения функции потерь для полученных предсказаний
    mse_error = loss_function(Y, prediction)

    # Округлить их до 0 и 1
    prediction[prediction >= 0.5] = 1
    prediction[prediction < 0.5] = 0
    prediction = prediction.astype(int)
    
    # Вычислить долю правильных отликов модели
    accuracy = np.mean(prediction == Y)

    if plot:
        plot_data(X, prediction)

    return mse_error, accuracy, prediction

def plot_data(X, Y):
    """
    Выводит данные на графике в соответствии со значениями
    
    Входные параметры:
        X (nd.array):   Образцы
        Y (nd.array):   Метки
    """
    colors = {}
    plt.figure("Данные")
    plt.scatter(X[:,0], X[:,1], color=np.array(["red" if x == 0 else "blue" for x in Y]))
    plt.show()


def plot_history(train_mse, train_accs, test_mse, test_accs):
    """
    Выводит изменения точности и потерь модели по обучающим и тестовым данным с течением эпох
    
    Входные параметры:
        train_mse (list):      Значения MSE на обучающей выборке
        train_accs (list):     Значения Accuracy на обучающей выборке
        test_mse (list):       Значения MSE на тестовой выборке
        test_accs (list):      Значения Accuracy на тестовой выборке
    """
    plt.figure("MSE")
    plt.plot([x[0] for x in train_mse], [x[1] for x in train_mse], label="MSE на обучающей выборке")
    plt.plot([x[0] for x in test_mse], [x[1] for x in test_mse], label="MSE на тестовой выборке")
    plt.legend()

    plt.figure("Accuracy")
    plt.plot([x[0] for x in train_accs], [x[1] for x in train_accs], label="Accuracy на обучающей выборке")
    plt.plot([x[0] for x in test_accs], [x[1] for x in test_accs], label="Accuracy на тестовой выборке")
    plt.legend()

    plt.show()

def save_pickle(obj, file_path):
    """
    Сохраняет объект в pickle-файл
    
    Входные параметры:
        obj (Object):       Объект для сохранения
        file_path (str):    Путь и имя файла
    """
    with open(file_path, "wb") as pckl:
        pickle.dump(obj, pckl, protocol=pickle.HIGHEST_PROTOCOL)

def load_pickle(file_path):
    """
    Загружает объект из pickle-файла
    
    Входные параметры:
        file_path (str):    Путь и имя файла
    Возвращаемое значение:
        obj (Object):       Объект для загрузки
    """
    with open(file_path, "rb") as pckl:
            obj =  pickle.load(pckl)
    return obj

def early_stopping(mses, n = 3):
    """
    Функция критерия ранней остановки, 
    проверяющая, уменьшилась ли MSE за последние n эпох.
    
    Входные параметры:
        mses (list):         Значения MSE
        n (int):             Число эпох для выполнения ранней остановки
    Возвращаемое значение:
        (bool):              Нужно останавливать обучение или нет
    """
    # Получить значение MSE
    mses = [x[1] for x in mses]

    # Проверить, было ли достигнуто наименьшее значение 
    # за последние n эпох; если да, продолжить обучение
    if min(mses) in mses[-n:]:
        return False
    else:
        print("Ранняя остановка!")
        return True
