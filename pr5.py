from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model, Sequential
import numpy as np
import csv
import pandas as pd


def gen_data(size):
    data = []
    targets = []
    for i in range(size):
        X = np.random.normal(3, 10)
        e = np.random.normal(0, 0.3)
        data.append((X ** 2 + e, np.sin(X / 2), np.cos(2 * X) + e, X - 3 + e, np.fabs(X) + e, (X ** 3) / 4 + e))
        targets.append((-X + e))
    return data, targets


train_data, train_targets = gen_data(300)
test_data, test_targets = gen_data(50)

pd.DataFrame(np.round(train_data, decimals=3)).to_csv("train_data.csv")
pd.DataFrame(np.round(train_targets, decimals=3)).to_csv("train_targets.csv")
pd.DataFrame(np.round(test_data, decimals=3)).to_csv("test_data.csv")
pd.DataFrame(np.round(test_targets, decimals=3)).to_csv("test_targets.csv")

mean = np.mean(train_data, axis=0)
train_data -= mean
std = np.std(train_data, axis=0)
train_data /= std

test_data -= mean
test_data /= std

main_input = Input(shape=(6,))
decode_input = Input(shape=(3,))

encoded = Dense(32, activation='relu')(main_input)
encoded = Dense(32, activation='relu')(encoded)
encoded = Dense(3, activation='relu')(encoded)

decoded = Dense(64, activation='relu', name='layer1')(encoded)
decoded = Dense(64, activation='relu', name='layer2')(decoded)
decoded = Dense(6, name='layer3')(decoded)

predicted = Dense(32, activation='relu')(encoded)
predicted = Dense(64, activation='relu')(predicted)
predicted = Dense(1)(predicted)

encoder = Model(main_input, encoded)
regression = Model(main_input, predicted)
auto_encoder = Model(main_input, decoded)

auto_encoder.compile(optimizer="adam", loss="mse", metrics=["mae"])
auto_encoder.fit(train_data, train_data, epochs=90,
                 batch_size=5, shuffle=True, validation_data=(test_data, test_data))
encode_data = encoder.predict(test_data)

decoder = auto_encoder.get_layer('layer1')(decode_input)
decoder = auto_encoder.get_layer('layer2')(decoder)
decoder = auto_encoder.get_layer('layer3')(decoder)
decoder = Model(decode_input, decoder)
decode_data = decoder.predict(encode_data)

regression.compile(optimizer="adam", loss="mse", metrics=['mae'])
regression.fit(train_data, train_targets, epochs=100,
               batch_size=10, validation_data=(test_data, test_targets))
predict_data = regression.predict(test_data)

regression.save('regression.h5')
decoder.save('decoder.h5')
encoder.save('encoder.h5')

pd.DataFrame(np.round(encode_data, decimals=3)).to_csv("encoded.csv")
pd.DataFrame(np.round(decode_data, decimals=3)).to_csv("decoded.csv")
pd.DataFrame(np.round(predict_data, decimals=3)).to_csv("predicted.csv")

