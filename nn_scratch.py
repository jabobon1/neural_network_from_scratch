from typing import Union, Optional
import mglearn
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


class Model:
    def __init__(self,
                 input_shape: int,
                 hidden_shape: int, output_shape: int):
        self.input_shape = input_shape
        self.hidden_shape = hidden_shape
        self.output_shape = output_shape

        self.weights_1 = np.random.uniform(-1, 1,
                                           (self.input_shape, self.hidden_shape))
        self.weights_2 = np.random.uniform(-1, 1, (self.hidden_shape, self.output_shape))
        self.bias_1 = np.random.uniform(-1, 1, (1, self.hidden_shape))
        self.bias_2 = np.random.uniform(-1, 1, (1, self.output_shape))

    @staticmethod
    def sigmoid(x: np.ndarray):
        """
        Сигмоидная функция
        1/(1+e^-x)
        :param x: float or np array of floats
        :return: float or np.array of floats
        """
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def sigmoid_derivative(x: Union[np.ndarray, np.float]):
        """
        Производная сигмоидной функции
        f(x)' = f(x) * (1-f(x))
        :param x: float or np array of float
        :return: float or np.array of float
        """
        sigmoid_out = Model.sigmoid(x)
        return sigmoid_out * (1.0 - sigmoid_out)

    @staticmethod
    def mean_square_error(y_true: Union[np.ndarray, np.float],
                          y_pred: Union[np.ndarray, np.float]):
        """
        Среднеквадратичная ошибка
        MSE = 1/n * Sum(y-y_pred)^2
        :param y_true: np.array of numbers
        :param y_pred: np.array of numbers
        :return: np.float
        """
        length = len(y_pred) if isinstance(y_pred, np.ndarray) else 1
        return 1 / length * np.sum(np.power(y_true - y_pred, 2))

    @staticmethod
    def mse_derivative(y_true: Union[np.ndarray, np.float],
                       y_pred: Union[np.ndarray, np.float]):
        """
        Производная функции среднеквадратичной ошибки
        MSE` =  1/n * sum(-2 * (y-y`))
        :param y_true: np.array of numbers
        :param y_pred: np.array of numbers
        :return: np.float
        """
        length = len(y_pred) if isinstance(y_pred, np.ndarray) else 1
        return 1 / length * np.sum(-2 * (y_true - y_pred))

    @staticmethod
    def accuracy(y_true: np.float,
                 y_pred: np.float):
        if y_true == np.round(y_pred):
            return 1
        return 0

    def predict(self, x: Union[np.ndarray, np.float]) -> Union[np.ndarray, np.float]:
        """
        Предсказание значений на основе входных данных
        Прямое прохождение через скрытые слои нейорсети
        :param x: input data
        :return: predicted values
        """
        try:
            if len(x.shape) == 1:
                assert x.shape[0] == self.input_shape
                return self._predict(x)
            elif len(x.shape) == 2:
                assert x.shape[1] == self.input_shape
                return np.array([self._predict(_x) for _x in x])
        except AssertionError:
            raise InputShapeError(f'Wrong shape of input data {x.shape}')

    def _predict(self, x: np.ndarray):
        # прохождение через первые веса от входных нейронов до скрытого слоя
        hidden_layer = np.dot(x, self.weights_1) + self.bias_1.squeeze()
        # выходные данные скрытого слоя после применения сигмоидальной функции
        out_hidden_layer = self.sigmoid(hidden_layer)
        # прохождение через последние веса к нейронам на последнеи сле
        output_layer = np.dot(self.weights_2.squeeze(), out_hidden_layer) + self.bias_2.squeeze()
        return self.sigmoid(output_layer)

    def update_weights(self,
                       x_train: np.ndarray,
                       y_true: np.float,
                       lr=0.001) -> (np.float, np.float):
        # FEEDFORWARDING то же самое, что в _predict()
        sum_h1 = np.dot(x_train, self.weights_1) + self.bias_1.squeeze()
        out_h1 = self.sigmoid(sum_h1)

        sum_h2 = np.dot(self.weights_2.squeeze(), out_h1) + self.bias_2.squeeze()
        out_h2 = self.sigmoid(sum_h2)

        # считаем производную среднеквадратичной ошибки
        output_mse_der = self.mse_derivative(y_true, out_h2)

        # выходноый слой
        # 2ой bias
        d_ypred_d_b2 = self.sigmoid_derivative(sum_h2)
        # 2ой скрытый слой
        d_ypred_d_w2 = np.dot(out_h1, d_ypred_d_b2)
        d_ypred_d_h1 = np.dot(self.weights_2, d_ypred_d_b2)

        # 1ый скрытый слой
        d_h1_d_b1 = self.sigmoid_derivative(sum_h1)

        d_h1_d_w1 = np.zeros(self.weights_1.shape)
        for i in range(self.hidden_shape):
            d_h1_d_w1[:, i] = np.dot(x_train, d_h1_d_b1[i])

        self.bias_1 -= lr * output_mse_der * \
                       d_ypred_d_h1.squeeze() * d_h1_d_b1.squeeze()
        self.weights_1 -= lr * output_mse_der * \
                          d_ypred_d_h1.squeeze() * d_h1_d_w1.squeeze()

        self.bias_2 -= lr * output_mse_der * d_ypred_d_b2
        self.weights_2 = (self.weights_2.squeeze() - lr *
                          output_mse_der * d_ypred_d_w2).reshape(-1, 1)

        return self.mean_square_error(y_true, out_h2), self.accuracy(y_true, out_h2)

    def fit(self,
            x: np.ndarray,
            y: np.ndarray,
            lr: float = 0.001,
            steps: int = 1000,
            early_stop_acc: Optional[float] = None,
            early_stop_loss: float = 0.18) -> np.ndarray:
        """
        Метод для обучения нейонной сети
        :param x: Входные данные
        :param y: Вызодные данные для обучения
        :param lr: learning rate
        :param steps: кол-во итераций
        :param early_stop_acc: опционально, если указано -
            обучение закончится по достижению нужных значений
        :param early_stop_loss: если указал early_stop_acc,
         обучение закончится по достижению нужных значений early_stop_acc и early_stop_loss

        :return: np.array of losses
        """
        losses = []
        for i in range(steps):
            print(f'ITERATION {i + 1}')
            t_loss = []
            accuracies = []
            for x_train, y_true in zip(x, y):
                loss, acc = self.update_weights(x_train, y_true, lr=lr)
                t_loss.append(loss)
                accuracies.append(acc)
            mean_loss = np.mean(t_loss)
            print('ACCURACY', np.mean(accuracies))
            print('LOSS', mean_loss)
            losses.append(mean_loss)
            if early_stop_acc and mean_loss < early_stop_loss and np.mean(
                    accuracies) >= early_stop_acc:
                break
        return np.array(losses)


class InputShapeError(Exception):
    pass


if __name__ == '__main__':
    X = np.array([[1, 1], [0, 0], [0, 1], [1, 0]])
    y = np.array([0, 0, 1, 1])

    plt.scatter([0, 1], [0, 1], marker='x')
    plt.scatter([0, 1], [1, 0], marker='o')
    plt.ylim(-0.25, 1.25)
    plt.legend(['zeroes', 'ones'], title='True values')
    plt.show()

    model = Model(input_shape=2, hidden_shape=3, output_shape=1)
    losses = model.fit(X, y, lr=0.1, steps=6000)
    plt.plot(losses)
    plt.show()

    # for x_train, y_true in zip(X, y):
    #     pred = model.predict(x_train)
    #     print(f'x_train: {x_train}\npredicted:{pred}\ntrue:{y_true}')
    #
    # cm2 = ListedColormap(['#619dff', '#8d5cce'])
    #
    # mglearn.plots.plot_2d_classification(classifier=model, X=X, fill=True, alpha=.4, cm=cm2)
    # mglearn.discrete_scatter(X[:, 0], X[:, 1], y)
    # plt.legend(['zeroes', 'ones'], title='True values')
    #
    # # mglearn.discrete_scatter(X_train[:, 0], X_train[:, 1], y_train)
    # plt.xlabel('X')
    # plt.ylabel('Y')
    # plt.show()
