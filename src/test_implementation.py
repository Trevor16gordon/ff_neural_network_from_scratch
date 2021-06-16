import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_digits
from main import FeedForwardNeuralNetwork

digits = load_digits()
X = np.array([im.flatten()/255 for im in digits.images])

def digits_to_array(num):
    ret = np.zeros(10)
    ret[num] = 1
    return ret

y = np.array([digits_to_array(x) for x in digits.target])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

x_len_input = 64
neurons_in_single_hidden_layer = 64
y_len_output = 10
ffnn = FeedForwardNeuralNetwork()
ffnn.fit(X_train, y_train, neurons_in_single_hidden_layer, n_iter=25)
y_test_hat = ffnn.predict(X_test)