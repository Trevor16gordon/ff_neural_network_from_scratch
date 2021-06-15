    """Trevor Gordon. 2021
    Implementation of a feed forward neural network from scratch.

    Drawing largely from https://dafriedman97.github.io/mlbook/content/c7/concept.html
    and https://www.youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi
    """


class FeedForwardNeuralNetwork():

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


    def fit(self, X, y, grad_step=1e-5, n_iter=1e3, seed=None):
        """Fit a feedforward neural network to the test set given.

        Args:
            X (np.array): Input data with columns for elements of a single input
            and rows for number of observations. X.shape = (num_observations, data_per_observation)
            y (np.array): Correct prediction for the input samples. y.shape = (data_per_output)
            grad_step (float, optional): Step for updating weights with gradient vectors. Defaults to 1e-5.
            n_iter (int, optional): Number of iterations. Defaults to 1e3.
        """
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