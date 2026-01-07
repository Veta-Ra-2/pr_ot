from data import load_heart_disease_dataset,load_heart_disease_experiment_dataset
from networks import NeuralNetwork
from layers import LinearLayer, ReLULayer, TanhLayer, SigmoidLayer, LeakyReLULayer, SwishLayer
from loss_functions import MSELoss
from auxiliary import calc_perf, plot_data, plot_history, early_stopping, load_pickle
import numpy as np
import kagglehub


path = kagglehub.dataset_download("akshaydattatraykhare/diabetes-dataset")
global dataset
dataset = path + '/diabetes.csv'

np.random.seed(0)

# Путь к директории с файлом модели
MODEL_PATH = "./models/"


def train_nn_model(inputs_n=2, outputs_n=1, hidden_layers_n=3,
                   hidden_neurons_n=[15, 25, 25], act_funcs=[2, 6, 3],
                   l_rate=0.0001, tr_epochs=500, early_stop_n=3,
                   save_weight=True, plot_hist=True, verbose=False):
    """
    Основная функция обучения модели (базовый вариант).
    """
    print("Обучение.")

    # Список слоев с функциями активации
    act_layers = [LinearLayer, ReLULayer, TanhLayer,
                  SigmoidLayer, LeakyReLULayer, SwishLayer]

    # Список используемых слоев согласно выбранным функциям активации
    used_layers = [act_layers[i - 1] for i in act_funcs]

    # Слить входы, выходы и число нейронов на каждом скрытом слое
    used_neurons = [inputs_n] + hidden_neurons_n + [outputs_n]

    # Создать список слоёв
    initialize_layers = [layer_c(used_neurons[i], used_neurons[i + 1])
                         for i, layer_c in enumerate(used_layers)]

    # Добавить выходной слой (линейный)
    initialize_layers += [LinearLayer(used_neurons[-2], used_neurons[-1])]

    # Построить модель нейросети
    network = NeuralNetwork(layers=initialize_layers)

    # Данные
    X_train, Y_train, X_test, Y_test = load_heart_disease_dataset(dataset)

    # Loss
    MSE = MSELoss()

    train_mse, train_accs, test_mse, test_accs = [], [], [], []

    for epoch in range(tr_epochs):
        epoch_mse, epoch_acc = [], []

        for x, y in zip(X_train, Y_train):
            out_value = network.forward(x.reshape((2, 1)))

            label = 1 if out_value >= 0.5 else 0
            right = int(y == label)
            epoch_acc.append(right)

            mse_err = MSE(y, out_value)
            epoch_mse.append(mse_err)

            error = MSE.backward()
            network._backpropagation(error)
            network._update_weights(l_rate)

        train_mse.append((epoch, np.mean(epoch_mse)))
        train_accs.append((epoch, np.mean(epoch_acc)))

        if verbose and epoch % 10 == 0:
            print(f"[*Обучение*] Эпоха {epoch:4} | MSE: {train_mse[-1][1]:2.3f} | Accuracy: {train_accs[-1][1]:1.3f}")

        mse_err, accuracy, predictions = calc_perf(X_test, Y_test, network)
        test_mse.append((epoch, mse_err))
        test_accs.append((epoch, accuracy))

        if early_stopping(test_mse, n=early_stop_n):
            break

    if save_weight:
        network._save_model(MODEL_PATH)

    if plot_hist:
        plot_history(train_mse, train_accs, test_mse, test_accs)

    print("Обучение окончено!")

    return network


def test_nn_model(network=None):
    """
    Тестирование модели.
    """
    print("Тестирование.")
    X_test = load_pickle('./data/X_test.pickle')
    Y_test = load_pickle('./data/Y_test.pickle')

    if not network:
        network = NeuralNetwork(load_s_weights=True, model_fp=MODEL_PATH + "model.pickle")

    mse_err, accuracy, predictions = calc_perf(X_test, Y_test, network)

    print(f"Тестирование окончено с accuracy = {accuracy:.2f} и MSE = {mse_err:.2f}")


#####################################
# Новый экспериментальный функционал #
#####################################


class Layer:
    def __init__(self, input_dimension, output_dimension):
        self.layer = LinearLayer(input_dimension, output_dimension)
        self.function = None  # сюда будем задавать активацию

    def forward(self, x):
        z = self.layer.forward(x)
        return self.function(z) if self.function else z


# Активации
def relu(x):
    return np.maximum(0, x)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# Функция обучения и оценки
def train_and_evaluate(network, learning_rate=0.001, epochs=100):
    X_train, Y_train, X_test, Y_test = load_heart_disease_experiment_dataset(dataset)
    MSE = MSELoss()

    for epoch in range(epochs):
        for x, y in zip(X_train, Y_train):
            out_value = network.forward(x.reshape((4, 1)))

            mse_err = MSE(y, out_value)
            error = MSE.backward()
            network._backpropagation(error)
            network._update_weights(learning_rate)

    _, accuracy, _ = calc_perf(X_test, Y_test, network)
    return accuracy


def run_experiments():
    hidden_layer_sizes = [4, 8, 16]
    learning_rates = [0.01, 0.001, 0.0001]
    epochs = 1000

    best_configuration = None
    best_accuracy = 0

    for hidden_size in hidden_layer_sizes:
        for lr in learning_rates:
            print(f"\n[Эксперимент] hidden_size={hidden_size}, lr={lr}")

            input_layer = Layer(input_dimension=4, output_dimension=hidden_size)
            input_layer.function = relu

            hidden_layer = Layer(input_dimension=hidden_size, output_dimension=hidden_size // 2)
            hidden_layer.function = relu

            output_layer = Layer(input_dimension=hidden_size // 2, output_dimension=1)
            output_layer.function = sigmoid

            network = NeuralNetwork(layers=[input_layer.layer, hidden_layer.layer, output_layer.layer])

            accuracy = train_and_evaluate(network, learning_rate=lr, epochs=epochs)
            print(f"Accuracy = {accuracy:.4f}")

            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_configuration = (hidden_size, lr)

    print(f"\nЛучшая конфигурация: hidden_size={best_configuration[0]}, lr={best_configuration[1]} с accuracy={best_accuracy:.4f}")


if __name__ == "__main__":
    # Базовое обучение
    trained_network = train_nn_model()
    test_nn_model(trained_network)

    # Эксперименты
    run_experiments()
