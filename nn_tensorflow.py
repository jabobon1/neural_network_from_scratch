import numpy as np
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD

model = Sequential([
    Dense(3, input_shape=(2,), activation='sigmoid', bias_initializer='ones'),
    Dense(1, activation='sigmoid')
])

model.compile(loss='mse', optimizer=SGD(learning_rate=0.1), metrics=['accuracy'])
model.summary()

X = np.array([[1, 1], [0, 0], [0, 1], [1, 0]])
Y = np.array([0, 0, 1, 1])

model.fit(X, Y, batch_size=1, epochs=3000, )

pred = model.predict(X)
print('predicted:', pred)
print('y_true:', Y)
