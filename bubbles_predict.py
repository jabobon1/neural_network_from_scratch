import mglearn
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn.datasets._samples_generator import make_blobs

from nn_scratch import Model

centers, n_features = 2, 2
# generate 2d classification dataset
X, y = make_blobs(n_samples=500, centers=centers, n_features=n_features)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

STEP_COUNT = 1500
lr = 0.01

model = Model(input_shape=2, hidden_shape=2, output_shape=1)
losses = model.fit(X_train, y_train, lr=lr, steps=STEP_COUNT, early_stop_acc=1)
plt.plot(losses)
plt.show()

cm2 = ListedColormap(['#619dff', '#8d5cce'])

mglearn.plots.plot_2d_classification(classifier=model, X=X_test, fill=True, alpha=.4, cm=cm2)
mglearn.discrete_scatter(X_test[:, 0], X_test[:, 1], y_test)
# mglearn.discrete_scatter(X_train[:, 0], X_train[:, 1], y_train)
plt.xlabel('1st feature')
plt.ylabel('2nd feature')
plt.show()
