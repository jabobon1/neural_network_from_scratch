import mglearn
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


class InputShapeError(Exception):
    pass


class Model:
    def __init__(self,
                 input_shape: int,
                 hidden_shape: int, output_shape: int):
        self.input_shape = input_shape
        self.hidden_shape = hidden_shape
        self.output_shape = output_shape

        self.hidden_output_1 = np.zeros(self.hidden_shape)
        self.hidden_output_2 = np.zeros(self.output_shape)

        self.weights_1 = np.random.uniform(-1, 1,
                                           (self.input_shape, self.hidden_shape))
        self.weights_2 = np.random.uniform(-1, 1, (self.hidden_shape, self.output_shape))
        self.bias_1 = np.random.uniform(-1, 1, (1, self.hidden_shape))
        self.bias_2 = np.random.uniform(-1, 1, (1, self.output_shape))

    @staticmethod
    def softmax(x: np.ndarray):
        return np.exp(x) / np.sum(np.exp(x))

    @staticmethod
    def tanh(x: np.ndarray):
        return np.tanh(x)

    @staticmethod
    def sigmoid(x: np.ndarray):
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def sigmoid_derivative(x):
        """
        The first derivative of the sigmoid function wrt x
        """
        sigmoid_out = Model.sigmoid(x)
        return sigmoid_out * (1.0 - sigmoid_out)

    @staticmethod
    def mean_square_error(y_true, y_pred):
        """
        mse = 1/n * Sum(y-y`)^2
        :param y_true: np.array
        :param y_pred: np.array
        :return: float
        """
        length = len(y_pred) if isinstance(y_pred, np.ndarray) else 1
        return np.sum(np.power(y_true - y_pred, 2)) / length

    @staticmethod
    def mse_derivative(y_true, y_pred):
        # x^n = nx^(n-1)
        """
        mse` =  1/n * -2 (y-y`)
        :param y_true: np.array
        :param y_pred: np.array
        :return: float
        """
        length = len(y_pred) if isinstance(y_pred, np.ndarray) else 1
        return 1 / length * np.sum(-2 * (y_true - y_pred))

    @staticmethod
    def accuracy(y_true, y_pred):
        if y_true == np.round(y_pred):
            return 1
        return 0

    def predict(self, x: np.ndarray):
        if len(x.shape) == 1 and x.shape[0] != self.input_shape:
            print(x.shape)
            raise InputShapeError
        elif len(x.shape) == 2 and x.shape[1] != self.input_shape:
            print(x.shape)
            raise InputShapeError
        if len(x.shape) == 2:
            res = []
            for _x in x:
                res.append(self._predict(_x))
            return np.array(res)
        elif len(x.shape) == 1:
            return self._predict(x)
        raise InputShapeError

    def _predict(self, x: np.ndarray):
        hidden_layer = self.sigmoid(np.dot(x, self.weights_1) + self.bias_1.squeeze())
        return self.sigmoid(np.dot(self.weights_2.squeeze(), hidden_layer) + self.bias_2.squeeze())

    def update_weights(self,
                       x_train: np.ndarray,
                       y_true: np.float,
                       lr=0.001) -> (np.float, np.float):
        # FEEDFORWARDING
        sum_h1 = np.dot(x_train, self.weights_1) + self.bias_1.squeeze()
        out_h1 = self.sigmoid(sum_h1)

        sum_h2 = np.dot(self.weights_2.squeeze(), out_h1) + self.bias_2.squeeze()
        out_h2 = self.sigmoid(sum_h2)

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

    def fit(self, X, y, lr=0.001, steps=1000, early_stop_acc=None) -> np.ndarray:
        losses = []
        for i in range(steps):
            print(f'ITERATION {i + 1}')
            t_loss = []
            accuracies = []
            for x_train, y_true in zip(X, y):
                loss, acc = self.update_weights(x_train, y_true, lr=lr)
                t_loss.append(loss)
                accuracies.append(acc)
            mean_loss = np.mean(t_loss)
            print('ACCURACY', np.mean(accuracies))
            print('LOSS', mean_loss)
            losses.append(mean_loss)
            if early_stop_acc and mean_loss < 0.17 and np.mean(accuracies) >= early_stop_acc:
                break
        return np.array(losses)


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

    for x_train, y_true in zip(X, y):
        pred = model.predict(x_train)
        print(f'x_train: {x_train}\npredicted:{pred}\ntrue:{y_true}')

    cm2 = ListedColormap(['#619dff', '#8d5cce'])

    mglearn.plots.plot_2d_classification(classifier=model, X=X, fill=True, alpha=.4, cm=cm2)
    mglearn.discrete_scatter(X[:, 0], X[:, 1], y)
    plt.legend(['zeroes', 'ones'], title='True values')

    # mglearn.discrete_scatter(X_train[:, 0], X_train[:, 1], y_train)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.show()
