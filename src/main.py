    """Trevor Gordon. 2021
    Implementation of a feed forward neural network from scratch.

    Drawing largely from https://dafriedman97.github.io/mlbook/content/c7/concept.html
    and https://www.youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi
    """

import numpy as np

class FeedForwardNeuralNetwork():
    """Feed Forward Neural Network

    This FFNN is fixed to have one hidden layer. It is defined by weights self.W1 and self.W2 and
    biases self.c1 and self.c2
    """

    def __init__(self) -> None:
        """Init function

        Class variables:
            self.n_hidden (int): Number of neurons in single hidden layer
            self.f1 (func): Activation function to apply to hidden layer output before next layer
                Only supporting sigmoid right now
            self.loss_func (func): Error function for training.
                Only supporting RSS
        """
        self.n_hidden = 1
        self.f1 = self.sigmoid
        self.loss_func = self.rss

    def initialize_params(self, len_input, len_hidden, len_output):
        """Initialize weights and biases randomly.

        Args:
            len_input (int): Number of neurons in X input layer
            len_hidden (int): Number of neurons in hidden layer
            len_output (int): Number of neurons in y output layer

        Returns:
            weights_1 (np.array): Rows for neurons in hidden layer and columns for 
                neurons in X input layer.
            weights_2 (np.array): Rows for neurons in y output layer and columns for neurons in
                hidden layer
            biases_1 (np.array): Rows for neurons in hidden layer and 1 column
            biases_2 (np.array): Rows for neurons in output layer and 1 column
        """
        weights_1 = np.random.randn(len_hidden, len_input)/5
        biases_1 = np.random.randn(len_hidden, 1)/5
        weights_2 = np.random.randn(len_output, len_hidden)/5
        biases_2 = np.random.randn(len_output, 1)/5
        return weights_1, biases_1, weights_2, biases_2


    def fit(self, X, y, len_hidden, grad_step=1e-5, n_iter=1e3, seed=None):
        """Fit a feedforward neural network to the test set given.

        

        Args:
            X (np.array): Input data with columns for elements of a single input
            and rows for number of observations. X.shape = (num_observations, data_per_observation)
            y (np.array): Correct prediction for the input samples. y.shape = (num_observations, len_output)
            len_hidden (int): Number of neurons in input layer.
            grad_step (float, optional): Step for updating weights with gradient vectors. Defaults to 1e-5.
            n_iter (int, optional): Number of iterations. Defaults to 1e3.
        """
        len_input = X.shape[1]
        len_output = y.shape[1]
        num_observations = len(X)
        self.W1, self.c1, self.W2, self.c2 = self.initialize_params(len_input, len_hidden, len_output)

        for i in n_iter:
            # Adjust weights and biases in the direction of the negative gradient of the loss function
            
            dL_dW1, dL_dc1, dL_dW2, dL_dc2 = self.get_loss_gradient()

            self.W1 -= grad_step * dL_dW1
            self.W2 -= grad_step * dL_dW2
            self.c1 -= grad_step * dL_dc1
            self.c2 -= grad_step * dL_dc2

            self.update_network_internal_states()
        return True

    def predict(self, X_predict):
        """Predict output for given input after model has been fit.

        Args:
            X_predict (np.array): Input data with columns for elements of a single input. 
            and rows for each prediction. X.shape = (num_to_predict, data_per_observation)

        Returns:
            np.array: Predictions. y.shape = (data_per_output, num_to_predict)
        """
        return y_predict