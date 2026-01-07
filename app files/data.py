import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from auxiliary import save_pickle

DATA_PATH = "./data/"

def load_heart_disease_experiment_dataset(file_path):
    data = pd.read_csv(file_path)
    print("Исходные данные:")
    print(data.head())

    df = data.copy()

    # Признаки, где 0 = пропуск
    cols_with_missing = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
    df[cols_with_missing] = df[cols_with_missing].replace(0, np.nan)

    # Заполняем пропуски медианами
    for col in cols_with_missing:
        df[col] = df[col].fillna(df[col].median())

    # Убираем выбросы по IQR для всех числовых признаков (кроме Outcome)
    numeric_cols = df.drop(columns="Outcome").columns
    for col in numeric_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        df[col] = np.where(df[col] < lower, lower, df[col])
        df[col] = np.where(df[col] > upper, upper, df[col])

    # Масштабируем все признаки (кроме Outcome)
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(df.drop(columns="Outcome"))
    scaled_df = pd.DataFrame(scaled_features, columns=numeric_cols)
    scaled_df["Outcome"] = df["Outcome"].values

    # Берём 4 признака: Pregnancies, Glucose, BloodPressure, BMI
    X = scaled_df[["Pregnancies", "Glucose", "BloodPressure", "BMI"]].values
    Y = scaled_df["Outcome"].values

    # Разделяем на train/test
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.2, random_state=42
    )

    # Сохраняем
    save_pickle(X_train, DATA_PATH + "X_train.pickle")
    save_pickle(Y_train, DATA_PATH + "Y_train.pickle")
    save_pickle(X_test, DATA_PATH + "X_test.pickle")
    save_pickle(Y_test, DATA_PATH + "Y_test.pickle")

    # Визуализация: Glucose vs BMI
    plt.scatter(
        scaled_df[scaled_df["Outcome"] == 0]["Glucose"],
        scaled_df[scaled_df["Outcome"] == 0]["BMI"],
        c="g", marker="^", label="No Diabetes"
    )
    plt.scatter(
        scaled_df[scaled_df["Outcome"] == 1]["Glucose"],
        scaled_df[scaled_df["Outcome"] == 1]["BMI"],
        c="b", marker="s", label="Diabetes"
    )
    plt.legend()
    plt.xlabel("Glucose (scaled)")
    plt.ylabel("BMI (scaled)")
    plt.title("Glucose vs BMI после предобработки")
    plt.show()

    return X_train, Y_train, X_test, Y_test
def load_heart_disease_dataset(file_path):
    # Загружаем данные
    data = pd.read_csv(file_path)
    print("Исходные данные:")
    print(data.head())

    df = data.copy()

    X = df.iloc[:, [1, 5]].copy()
    Y = df["Outcome"].values

    # Заменяем 0 на NaN (только для выбранных признаков)
    X = X.replace(0, np.nan)

    # Заполняем пропуски медианой
    for col in X.columns:
        X[col] = X[col].fillna(X[col].median())

    # Убираем выбросы по IQR для этих 2 признаков
    for col in X.columns:
        Q1 = X[col].quantile(0.25)
        Q3 = X[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        X[col] = np.where(X[col] < lower, lower, X[col])
        X[col] = np.where(X[col] > upper, upper, X[col])

    # Переводим в numpy
    X = X.values

    # Разделяем на train/test
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.2, random_state=42
    )

    # Масштабируем (StandardScaler)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Сохраняем
    save_pickle(X_train, DATA_PATH + "X_train.pickle")
    save_pickle(Y_train, DATA_PATH + "Y_train.pickle")
    save_pickle(X_test, DATA_PATH + "X_test.pickle")
    save_pickle(Y_test, DATA_PATH + "Y_test.pickle")

    # Визуализация (Glucose vs BMI)
    plt.scatter(X[Y == 0][:, 0], X[Y == 0][:, 1], c="g", marker="^", label="No Diabetes")
    plt.scatter(X[Y == 1][:, 0], X[Y == 1][:, 1], c="b", marker="s", label="Diabetes")
    plt.legend()
    plt.xlabel("Glucose")
    plt.ylabel("BMI")
    plt.title("Диаграмма после предобработки")
    plt.show()

    return X_train, Y_train, X_test, Y_test

'''Путь к директории с данными
DATA_PATH = "./data/"

def generate_dataset(N=1000, c_val=0.5, r=1/np.sqrt(2*np.pi)):
    """
    Функция для генерации обучающих и тестовых данных
    Точки генерируются равномерно в интервале [0,1]². 
    Им будет присвоена метка 1, если они находятся внутри круга с радиусом r и центром c_val
    Входные параметры:
        N (int):         число точек (экземпляров, образцов) данных
        c_val (float):   центр круга
        r (float):       радиус круга
    Возвращаемые значения:
        X_train (np.ndarray [Nx2]):     обучающие образцы
        Y_train (np.ndarray [Nx1]):     метки к обучающим образцам
        X_test (np.ndarray [Nx2]):      тестовые образцы 
        Y_test (np.ndarray [Nx1]):      метки к тестовым образцам
    """
    # Случайным образом сгенерировать обучающие образцы набора данных
    X = np.random.uniform(0, 1, (N * 2, 2))
    
    # Получить метки: вычесть центр и вычислить норму
    Y = (np.linalg.norm(X - c_val, axis=1) <= r).astype(int)
    Path(DATA_PATH).mkdir(exist_ok=True)
    
    # Разбить данные на обучающие и тестовые и сохранить в pickle-файлах
    save_pickle(X[:N], DATA_PATH + "X_train.pickle")
    save_pickle(Y[:N], DATA_PATH + "Y_train.pickle")
    save_pickle(X[N:], DATA_PATH + "X_test.pickle")
    save_pickle(Y[N:], DATA_PATH + "Y_test.pickle")
    
    # Вывести данные на графике
    #plt.plot(X[:,0][Y == 0], X[:, 1][Y == 0], 'g^')  # зеленые треугольники
    #plt.plot(X[:,0][Y == 1], X[:, 1][Y == 1], 'bs')  # синие квадраты
    #plt.show()
    
    # Вернуть обучающие и тестовые выборки и метки к экземплярам данных
    return X[:N], Y[:N], X[N:], Y[N:]
'''
