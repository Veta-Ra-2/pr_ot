import matplotlib.pyplot as plt
from matplotlib import style
import matplotlib
matplotlib.use( 'tkagg' )
from tkinter import *
from test_model import *
from tkinter import messagebox as mb
    
def check_inputs(hid_layers_n, hid_neurons_n, act_funcs, l_rate, num_tr_epochs):
    """
    Для всех переменных, определенных пользовательским вводом, проверяет правильность значений.
    Входные параметры:
        hid_layers_n (int):     Пользовательский ввод для числа скрытых слоев
        hid_neurons_n (list):   Пользовательский ввод для числа нейронов на каждом скрытом слое
        act_funcs (list):       Пользовательский ввод для функции активации на каждом скрытом слое
        l_rate (float):         Пользовательский ввод для скорости обучения
        num_tr_epochs (int):    Пользовательский ввод для числа эпох обучения
    """
    if not (hid_layers_n == len(hid_neurons_n)):
        mb.showerror("Ошибка", f"Число нейронов должно быть задано для каждого слоя! Задано для ({len(hid_neurons_n)}) слоев, а должно быть ({hid_layers_n}) слоев.")
        return False
    if not (min(hid_neurons_n) > 0):
        mb.showerror("Ошибка", f"Число нейронов должно быть > 0, а не {min(hid_neurons_n)}!")
        return False
    if not (hid_layers_n == len(act_funcs)):
        mb.showerror("Ошибка", f"Функции активации должны быть заданы для каждого слоя! Задано для ({len(hid_neurons_n)}), а должно быть ({hid_layers_n}) функций активации.")
        return False
    if not (max(act_funcs) <= 6 and min(act_funcs) >= 1):
        problem_number = max(act_funcs) if max(act_funcs) > 6 else min(act_funcs)
        mb.showerror("Ошибка", f"Функции активации должны быть заданы числами (1,2,3,4,5,6), а не как {problem_number}.")
        return False
    if not (l_rate > 0):
        mb.showerror("Ошибка", f"Скорость обучения должна быть > 0, а не {l_rate}.")
        return False
    if  not (num_tr_epochs > 0):
        mb.showerror("Ошибка", f"Число эпох должно быть > 0, а не {num_tr_epochs}.")
        return False
    return True

def main_window():
    """
    Графический пользовательский интерфейс - окно для ввода параметров модели
    """
    def submit():
        """
        Событие, возникающее при нажатии кнопки "Запуск"
        """
        try:
            # Получить значения из элементов формы
            hid_layers_n = int(hid_layers_entry.get())
            hid_neurons_n = [int(x) for x in list(hid_neurons_entry.get().split(","))]
            act_funcs = [int(x) for x in list(act_funcs_entry.get().split(","))]
            l_rate = float(lr_entry.get())
            num_tr_epochs = int(epochs_entry.get())
        except ValueError:
            mb.showerror(f"Ошибка ввода!", "В текстовые поля необходимо вводить числа в указанном формате")
            return

        # Проверить ошибки пользовательского ввода (в случае ошибок выйти)
        if not check_inputs(hid_layers_n, hid_neurons_n,
                        act_funcs, l_rate, num_tr_epochs):
            return

        if True:
            # Закрыть окно программы
            root.destroy()

            # Выполнить обучение с числом входов 2 (координаты точек) и выходов 1 (бинарная классификация)
            trained_network = train_nn_model(2, 1, hid_layers_n, hid_neurons_n, act_funcs, l_rate, num_tr_epochs, verbose = True)
            test_nn_model(trained_network)

    # Создать окно программы
    root = Tk()
    root.geometry("1000x500")
    root.title("Многослойный персептрон")

    # Создать метку
    label_head = Label(text = "Гиперпараметры сети", bg = "grey", fg = "black", width = "500", height = "3")
    label_head.pack()

    # Создать метки
    hid_layers_text = Label(root, text="Количество скрытых слоев:")
    hid_neurons_text = Label(root, text="Список количества нейронов на каждом скрытом слое:")
    act_funcs_text = Label(root, text="Список функций активаций на скрытых слоях (линейная = 1, ReLU = 2, гип. тангенс = 3, сигмоида = 4, ReLU с утечкой = 5, Swish = 6):")
    lr_text = Label(root, text="Скорость обучения:")
    epochs_text = Label(root, text="Количество эпох:")
    
    # Разместить метки на форме
    hid_layers_text.place(x = 15, y = 70)
    hid_neurons_text.place(x = 15, y = 140)
    act_funcs_text.place(x = 15, y = 210)
    lr_text.place(x = 15, y = 280)
    epochs_text.place(x = 15, y = 350)

    # Создать поля для ввода значений
    hid_layers_entry = Entry(root, width = "120")
    hid_neurons_entry = Entry(root, width = "120")
    act_funcs_entry = Entry(root, width = "120")
    lr_entry = Entry(root, width = "120")
    epochs_entry = Entry(root, width = "120")
    
    # Ввести значения по умолчанию в поля ввода
    hid_layers_entry.insert(0, "3")
    hid_neurons_entry.insert(0, "10, 15, 25")
    act_funcs_entry.insert(0, "4, 6, 5")
    lr_entry.insert(0, "0.0001")
    epochs_entry.insert(0, "300")
    
    # Разместить поля на форме
    hid_layers_entry.place(x = 15, y = 100)
    hid_neurons_entry.place(x = 15, y = 170)
    act_funcs_entry.place(x = 15, y = 240)
    lr_entry.place(x = 15, y = 310)
    epochs_entry.place(x = 15, y = 380)

    # Создать кнопку для запуска программы
    do_button = Button(root, text = 'Запуск', command = submit)
    do_button.place(x = 480, y = 430)

    # Запустить окно
    root.mainloop()

if __name__=="__main__":
    main_window()
    input()