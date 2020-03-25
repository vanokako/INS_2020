from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
import numpy as np
from math import exp
import matplotlib.pyplot as plt

def element_wise_predict(data, weights):
    relu = lambda x: max(x, 0)
    sigmoid = lambda x: 1 / (1 + exp(-x))
    act = [relu for _ in weights]
    act[-1] = sigmoid
    tmp_data = data.copy()
    for d in range(len(weights)):
        res = np.zeros((tmp_data.shape[0], weights[d][0].shape[1]))
        for i in range(tmp_data.shape[0]):
            for j in range(weights[d][0].shape[1]):
                s = 0
                for k in range(tmp_data.shape[1]):
                    s += tmp_data[i][k] * weights[d][0][k][j]
                res[i][j] = act[d](s + weights[d][1][j])
        tmp_data = res
    return res

def tensor_predict(input_data, weights):
    relu = lambda x: np.maximum(x, 0)
    sigmoid = lambda x: 1/(1+np.exp(-x))
    act = [relu for _ in weights]
    act[-1] = sigmoid
    res = input_data.copy()
    for d in range(0, len(weights)):
        res = act[d](np.dot(res, weights[d][0]) + weights[d][1])
    return res

def print_predicts(model, dataset):
    weights = []
    for layer in model.layers:
        weights.append(layer.get_weights())
    element_wise_res = element_wise_predict(dataset, weights)
    tensor_res = tensor_predict(dataset, weights)
    model_res = model.predict(dataset)
    assert np.isclose(element_wise_res, model_res).all()
    assert np.isclose(tensor_res, model_res).all()
    print('Element')
    print(element_wise_res)
    print('Tensor')
    print(tensor_res)
    print('Model')
    print(model_res)

def logic_func(a, b, c):
    return (a != b) and (b != c)


train_data = np.array([[0, 0, 0],
                        [0, 0, 1],
                        [0, 1, 0],
                        [0, 1, 1],
                        [1, 0, 0],
                        [1, 0, 1],
                        [1, 1, 0],
                        [1, 1, 1]])
train_target = np.array([int(logic_func(*x)) for x in train_data])
model = Sequential()
model.add(Dense(16, activation='relu', input_shape=(3,)))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='adam',loss='binary_crossentropy', metrics=['accuracy'])

print_predicts(model, train_data)
model.fit(train_data, train_target, epochs=150, batch_size=1)
print_predicts(model, train_data)