import numpy as np
from datetime import datetime
import tensorflow.keras as keras
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.models import Sequential, load_model, save_model
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

from tensorflow.keras.utils import to_categorical

from sklearn.model_selection import train_test_split

from var6 import gen_data

class my_callback(Callback):
    def __init__(self, val, prefix='my_model_number', key='val_loss', date=datetime.now()):
        self.val = val
        self.prefix='{}_{}_{}_{}_'.format(date.day, date.month, date.year, prefix)
        self.loss = {}
        self.key = key
        self.index = 0


    def on_train_begin(self, logs=None):
        loss = self.model.evaluate(self.val[0], self.val[1])[0]
        for i in range(1,4):
            self.loss[self.prefix + str(i)] = loss
        for key in self.loss.keys():
            self.model.save(key)

    def on_epoch_end(self, epoch, logs=None):
        for i in range(1, 4):
            if logs.get(self.key) < self.loss[self.prefix + str(i)] and i > self.index:
                self.loss[self.prefix + str(i)] = logs.get(self.key)
                self.model.save(self.prefix + str(i))
                self.index += 1
                break
            elif i <= self.index:
                continue
        if (self.index == 3):
            self.index = 0


    def on_train_end(self, logs=None):
        for (key, loss) in self.loss.items():
            print(key + ' ' + str(loss))

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
history = model.fit(x_train, y_train, epochs=4, batch_size=100,
              validation_data=(x_test, y_test), callbacks=[my_callback((x_test, y_test))])
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