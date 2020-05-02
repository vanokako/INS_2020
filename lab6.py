import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow.keras.datasets import imdb

dimensions = [500, 1000, 2500, 5000, 7500]
accuracy = []

def vectorize(sequences, dimension=10000):
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1
    return results

def load_data(dimension):
    (training_data, training_targets), (testing_data, testing_targets) = imdb.load_data(num_words=dimension)
    data = np.concatenate((training_data, testing_data), axis=0)
    targets = np.concatenate((training_targets, testing_targets), axis=0)
    data = vectorize(data, dimension)
    targets = np.array(targets).astype("float32")
    test_x = data[:10000]
    test_y = targets[:10000]
    train_x = data[10000:]
    train_y = targets[10000:]
    return (train_x, train_y), (test_x, test_y)

def build_model():
    model = models.Sequential()
    model.add(layers.Dense(50, activation="relu"))
    model.add(layers.Dropout(0.3, noise_shape=None, seed=None))
    model.add(layers.Dense(50, activation="relu"))
    model.add(layers.Dropout(0.2, noise_shape=None, seed=None))
    model.add(layers.Dense(50, activation="relu"))
    model.add(layers.Dense(1, activation="sigmoid"))
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model

def test_dims(dimension):
    (train_x, train_y), (test_x, test_y) = load_data(dimension)
    model = build_model()
    history = model.fit(train_x, train_y, epochs=2, batch_size=500, validation_data=(test_x, test_y))
    accuracy.append(history.history['val_acc'])


def load_text(filename):
    file = open(filename, 'rt')
    text = file.read()
    file.close()
    words = text.split()
    import string
    table = str.maketrans('', '', string.punctuation)
    stripped = [w.translate(table) for w in words]
    stripped_low = []
    for w in stripped:
        stripped_low.append(w.lower())
    print(stripped_low)
    indexes = imdb.get_word_index()
    encoded = []
    for w in stripped_low:
        if w in indexes and indexes[w] < 7500:
            encoded.append(indexes[w])
    data = np.array(encoded)
    (train_x, train_y), (test_x, test_y) = load_data(7500)
    model = build_model()
    history = model.fit(train_x, train_y, epochs=2, batch_size=500,
                        validation_data=(test_x, test_y))
    data = vectorize([data], 7500)
    res = model.predict(data)
    print(res)

load_text('text.txt')
# for dimension in dimensions:
#     test_dims(dimension)

# plt.plot(dimensions, accuracy,  label='Validation acc')
# plt.title('Validation accuracy for different dimensions')
# plt.xlabel('Dimensions')
# plt.ylabel('Accuracy')
# plt.legend()
# plt.show()
