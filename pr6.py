import numpy as np

import tensorflow.keras as keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

from tensorflow.keras.utils import to_categorical

from sklearn.model_selection import train_test_split

from var6 import gen_data

X, Y = gen_data(1000)
X = np.asarray(X)

Y = np.asarray(Y).flatten()
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
encoder = LabelEncoder()
encoder.fit(Y)
y_test = np.asarray(encoder.transform(y_test))
y_train = np.asarray(encoder.transform(y_train))
y_test = to_categorical(y_test)
y_train = to_categorical(y_train)

x_train = np.asarray(x_train).reshape(800, 50, 50, 1)
x_test = np.asarray(x_test).reshape(200, 50, 50, 1)

model = Sequential()
model.add(Conv2D(32, kernel_size=5, activation='relu', input_shape=(50, 50, 1)))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Conv2D(64, kernel_size=5, strides=1, activation='relu'))
model.add(Conv2D(64, kernel_size=5, strides=1, activation='relu'))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(3, activation='softmax'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
history = model.fit(x_train, y_train, epochs=10, batch_size=100,
              validation_data=(x_test, y_test))
model.evaluate(x_test, y_test)

loss = history.history['loss']
val_loss = history.history['val_loss']
acc = history.history['acc']
val_acc = history.history['val_acc']
epochs = range(1, len(loss) + 1)
plt.plot(epochs, loss, 'r', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()
plt.clf()


plt.plot(epochs, acc, 'r', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()