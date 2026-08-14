import tensorflow as tf
from tensorflow.keras import Sequential, layers,models,optimizers,initializers
import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import mnist

#Important : In my file all np.ndarrays are dim : (n_sample_feaures by batch_size)

#################################
#MY_DATA : Import and preprocessing
#################################

def get_MNIST_data():
    """
    Imports the MNIST dataset, scales pixels to [0, 1], and reshapes images.

    Returns:
        Tuple: (X_train, Y_train, X_val, Y_val) where X shapes are (784, n_samples)
        and Y shapes are (10, n_samples) one-hot encoded.
    """
    (X_train, Y_train), (X_val, Y_val) = mnist.load_data()
    X_train = X_train.reshape(-1, 784).T
    X_val = X_val.reshape(-1, 784).T
    Y_train = en_code_onehot(Y_train)
    Y_val = en_code_onehot(Y_val)
    X_train = X_train / 255
    X_val = X_val / 255
    return (X_train, Y_train, X_val, Y_val)

def de_code_to_digit(X):
    """
    Decodes one-hot encoded label columns back into scalar digits.

    Args:
        X: One-hot encoded dataset of shape (10, n_samples).

    Returns:
        list: A list of integer digits of length n_samples.
    """
    digits = []
    for point in X.T:
        digits.append(int(np.argmax(point)))
    return digits

def en_code_onehot(X):
    """
    Encodes a list of integer digits into one-hot vectors.

    Args:
        X: List or array of integer digits.

    Returns:
        np.ndarray: One-hot encoded matrix of shape (10, n_samples).
    """
    one_hot_coding = np.zeros((10,len(X)))
    for i,digit in enumerate(X):
        one_hot_coding[digit][i] = 1
    return one_hot_coding

#################################
#FUNCTIONS : Activation functions and derivatives of the loss_function
#################################

def backward_loss(function,x,y,epsilon = "optional"):
    """
    Computes the derivative of the loss function with respect to the network output.

    Args:
        function: Name of the loss function ("NLL", "MSE", "Hinge").
        x: The output of the neural network.
        y: The target labels.
        epsilon: Optional threshold parameter for Hinge loss.

    Returns:
        np.ndarray: dLoss/dOutput. For Cross Entropy ("NLL"), it returns dLoss/dZ
        for mathematical simplification.
    """
    if(function == "NLL"):
        #Categorical Cross Entropy/Binary Cross Entropy
        return x - y
    if(function == "MSE"):
        #Mean Squared Error
        return np.mean((x-y)**2)
    if(function == "Hinge"):
        #Hinge Loss
        return np.where(abs(y - x) - epsilon > 0, -np.sign(y - x), 0.0)

#Basic activation functions : tanh, linear, relu, sigmoid, softmax
#Backward : The derivative of the function df/dx, forward : The value f(x)
#Backward for softmax is not needed, because with softmax I would use Categorical Cross Entropy
#which will simplify to nice mathematical dLossdZ = x - y

def forward_tanh(x):
    return np.tanh(x)
def backward_tanh(x):
    return 1 - forward_tanh(x) ** 2
def forward_linear(x):
    return x
def backward_linear(x):
    return 1
def forward_sigmoid(x):
    return 1 / (1 + np.exp(-x))
def backward_sigmoid(x):
    return forward_sigmoid(x) * (1 - forward_sigmoid(x))
def forward_ReLU(x):
    return np.maximum(x, 0)
def backward_ReLU(x):
    return np.where(x > 0, 1.0, 0.0)
def forward_softmax(x):
    x = x - np.max(x,axis = 0,keepdims = True)
    return np.exp(x)/np.sum(np.exp(x),axis = 0,keepdims = True)

##############################
#CLASSES Learned_parameter, Function, and function for drawing the accuracy plot
##############################
class Learned_parameter:
    """
    Represents a trainable parameter of the neural network (e.g., weights or biases).

    Attributes:
        data: The current value of the parameter.
        grad: The gradient with respect to this parameter, used in the Adam optimizer.
    """
    def __init__(self,data,grad):
        self.data = data
        self.grad = grad
class Function:
    '''
    Represents activation function, forward is this function, backward is its derivative
    '''
    def __init__(self,forward_function,backward_function):
        self.forward_function = forward_function
        self.backward_function = backward_function

#My basic functions created as instance of a Function
ReLU = Function(forward_ReLU,backward_ReLU)
sigmoid = Function(forward_sigmoid,backward_sigmoid)
tanh = Function(forward_tanh,backward_tanh)
softmax = Function(forward_softmax,None)

def draw_accuracy(accuracy,epochs):
    """
    Plots validation accuracy over training epochs.

    Args:
        accuracy: List of accuracy values after each epoch.
        epochs: Range or list of epoch numbers.

    Returns:
        None (displays the plot).
    """
    plt.plot(epochs, accuracy, marker="o", color="b", label="Accuracy")
    plt.xlabel("Epoka")
    plt.ylabel("Accuracy")
    plt.title("Dokładność modelu w kolejnych epokach")
    plt.grid(True)
    plt.legend()
    plt.show()

##############################
#MY LAYERS
##############################

class Module :
    '''
    Parent class for every module
    '''
    def __init__(self):
        pass
    def forward(self,X,training = None):
        raise NotImplementedError
    def backward(self,X):
        raise NotImplementedError

class Linear(Module):
    """
    Represents a fully connected (linear) layer in the network.

    Attributes:
        weights: Learned_parameter instance for the weight matrix.
        biases: Learned_parameter instance for the bias vector.
        Z: The output of this module.
        X: The input to this linear module.
    """
    def __init__(self,dim_input,dim_output):
        #Inicialization He
        self.weights = Learned_parameter(np.random.normal(0, np.sqrt(2) * (1/dim_input ** (.5)), [dim_input,dim_output]),0)
        self.biases = Learned_parameter(np.zeros((dim_output,1)),0)
        self.Z = np.zeros((dim_output,1))
    def give_params(self):
        '''
        :return: Returns list of learned params, necessary for Adam computation
        '''
        return[self.weights,self.biases]
    def forward(self,X,training = None):
        """
        Computes the linear transformation (forward pass).

        Args:
            X: Input array of shape (dim_input, batch_size).
            training: Unused parameter, kept for interface consistency across modules.

        Returns:
            np.ndarray: The output matrix Z of shape (dim_output, batch_size).
        """
        self.X = X
        self.Z = self.weights.data.T@X + self.biases.data
        return self.Z
    def backward(self,dLossdZ):
        """
        Performs the backward pass (backpropagation) for the linear module.

        Args:
            dLossdZ: Gradient with respect to the output of shape (dim_output, batch_size).

        Returns:
            np.ndarray: Gradient with respect to the input of shape (dim_input, batch_size).
        """
        self.weights.grad = self.X@dLossdZ.T
        self.biases.grad = np.sum(dLossdZ,axis = 1,keepdims = True)
        dLossdX = self.weights.data @ dLossdZ
        return dLossdX

class Activation(Module):
    """
    Represents the activation module of the neural network.

    Attributes:
        function: The activation function object.
        Z: The input cached from the forward pass.
    """
    def __init__(self,function):
        self.function = function

    def forward(self,Z,training = None):
        """
        Computes the activation function forward pass.

        Args:
            Z: The input from the linear module of shape (number_of_neurons, batch_size).
            training: Unused parameter, kept for interface consistency.

        Returns:
            np.ndarray: The activated output of shape (number_of_neurons, batch_size).
        """
        self.Z = Z
        A = self.function.forward_function(Z)
        return A
    def backward(self,dLossdA):
        """
        Performs the backward pass for the activation module.

        Args:
            dLossdA: Gradient with respect to the output of shape (number_of_neurons, batch_size).

        Returns:
            np.ndarray: Gradient with respect to the input of shape (number_of_neurons, batch_size).
        """
        dLossdZ = dLossdA * self.function.backward_function(self.Z)
        return dLossdZ
class Batch_Norm(Module):
    """
    Represents a Batch Normalization layer, which normalizes inputs to stabilize and speed up training.

    Attributes:
        G: Learned scaling parameter
        B: Learned shifting parameter 
        running_mean: Moving average of the mean, used during inference.
        running_var: Moving average of the variance, used during inference.
        K: Batch size stored from the forward pass.
        Z_normed: Cached normalized values before scaling and shifting.
    """
    def __init__(self,n):
        self.G = Learned_parameter(np.ones((n,1)),0)
        self.B = Learned_parameter(np.zeros((n,1)),0)
        self.running_mean = 0
        self.running_var = 0

    def give_params(self):
        '''
        :return: Returns learning params, necessary for computing Adam step size
        '''
        return [self.G,self.B]
    def forward(self,Z,training = None):
        """
        Computes the forward pass for Batch Normalization.

        Args:
            Z: The input to this layer, typically the output of a linear module.
            Shape: (number_of_neurons, batch_size).
            training: Boolean flag. During training, computes mean and variance for the current batch
                      and updates running averages. During inference (prediction), uses the stored
                      running averages to handle single points or arbitrary batches.

        Returns:
            np.ndarray: The output of this module. Shape: (number_of_neurons, batch_size).
        """
        self.K = Z.shape[1]
        if training :
            #We count mean and var of each feature and than normalize each of the features
            mean = np.mean(Z, axis=1, keepdims=True)
            var = np.var(Z,axis = 1,keepdims = True)
            self.running_mean = 0.9 * self.running_mean + 0.1 * mean
            self.running_var = 0.9 * self.running_var + 0.1 * var
            self.sd = np.sqrt(var)
            self.Z_normed = (Z - mean) / (self.sd + 1e-3)
        else:
            #We base on running averages counted while training
            self.Z_normed = (Z - self.running_mean)/(np.sqrt(self.running_var)+1e-3)
        #The normalized values are scaled and biased
        Z_final = self.Z_normed*self.G.data + self.B.data
        return Z_final

    def backward(self,dLossdA):
        """
        Performs the backward pass for the Batch Normalization layer. Typically placed before Activation module

        Args:
            dLossdA: Gradient with respect to the output of shape (number_of_neurons, batch_size).

        Returns:
            np.ndarray: Gradient with respect to the input (linear module output) of shape (number_of_neurons, batch_size).
        """
        #We compute dLossdB and dLossdG
        self.B.grad = np.sum(dLossdA,axis = 1,keepdims = True)
        self.G.grad = np.sum(dLossdA*self.Z_normed,axis = 1,keepdims = True)
        #We compute dLossdZ
        dLossdZ_norm = dLossdA * self.G.data
        dLossdZ = (1/self.K * 1/np.sqrt(self.sd**2 + 1e-3)*
                   (self.K*dLossdZ_norm - np.sum(dLossdZ_norm,axis = 1,keepdims = True)
                   -self.Z_normed*np.sum(dLossdZ_norm*self.Z_normed, axis = 1, keepdims = True)))
        return dLossdZ
class Dropout(Module):
    """
    Represents a Dropout module used for regularization to prevent overfitting.
    Typically placed after the activation function.

    Attributes:
        p: The probability of setting any feature value to zero during training.
    """
    def __init__(self,p):
        self.p = p
    def forward(self,A,training = True):
        """
        Computes the forward pass for the Dropout layer.

        Args:
            A: The input from the activation function of shape (number_of_neurons, batch_size).
            training: Boolean flag. If True, randomly zeroes out elements with probability p and scales the remaining values.
                      If False (during inference), passes the input through unchanged.

        Returns:
            np.ndarray: The output of the dropout layer of shape (number_of_neurons, batch_size).
        """
        n = A.shape[0]
        m = A.shape[1]
        if training :
            self.mask = np.random.choice([0, 1], size=(n, m), p=[0.2, 0.8])
            #We scale not to loose the total sum of information
            self.mask = self.mask/(1-self.p)
            A = A* self.mask
        return A
    def backward(self,dLossdZ):
        """
        Performs the backward pass for the Dropout layer.

        Args:
            dLossdZ: Gradient with respect to the output of shape (number_of_neurons, batch_size).

        Returns:
            np.ndarray: Gradient with respect to the activation module output of shape (number_of_neurons, batch_size).
        """
        return dLossdZ * self.mask
class Adam:
    """
    Represents the Adam optimizer.

    Default values:
        - 0.9 for the first moment running average (m)
        - 0.999 for the second moment running average (v)
        - 0.001 for the learning rate
        - 1e-8 for epsilon (to prevent division by zero)

    Attributes:
        lr_rate: The step size used for parameter updates.
        parameters: References to each learnable parameter in the network.
        m: Moving average of the gradients (first moment) for each parameter.
        v: Moving average of the squared gradients (second moment) for each parameter.
        t: Step counter tracking the number of update iterations performed.
    """
    def __init__(self,parameters,lr_rate):
        self.lr_rate = lr_rate
        self.parameters = parameters
        self.m = [0]*len(parameters)
        self.v = [0]*len(parameters)
        self.t = 0
    def step(self):
        """
        Performs a single optimization step using the Adam update rule,
        updating the model parameters and moving averages.
        """
        self.t += 1
        for i,param in enumerate(self.parameters):
            self.m[i] = self.m[i] * 0.9 + 0.1*param.grad
            self.v[i] = self.v[i] * 0.999 + 0.001*param.grad**2
            m_new = self.m[i]/(1-0.9**self.t)
            v_new = self.v[i]/(1-0.999**self.t)
            param.data = param.data - self.lr_rate * m_new / np.sqrt(v_new + 1e-8)

####################################
#NEURAL NETWORK CLASS
####################################

class Net:
    def __init__(self,list_of_modules,loss_function,metrics,plot,lr_rate):
        """
        Represents the entire neural network model.

        Attributes:
            modules: List of network layers/modules.
            loss_function: The loss function used to evaluate model performance.
            L: Total number of modules in the network.
            metrics: Accuracy metrics to be computed during training.
            accuracy_history: Historical accuracy data used for plotting (epochs on the x-axis).
            plot: Boolean flag indicating whether to generate a training plot.
            lr_rate: Learning rate, necessary for the optimizer.
            list_of_parameters: List of learnable parameters updated during training.
            optimizer: The optimization algorithm used (e.g., Adam).
            net_output: The final output of the entire network.
        """
        self.modules = list_of_modules
        self.loss_function = loss_function
        self.L = len(list_of_modules)
        self.metrics = metrics
        self.accuracy_history = []
        self.plot = plot
        self.lr_rate =lr_rate
        self.list_of_parameters = self.get_parameters()
        self.optimizer = Adam(self.list_of_parameters,lr_rate)

    def get_parameters(self):
        """
        Gathers all learnable parameters from every module in the neural network.

        Returns:
            list: A combined list containing references to all model parameters.
        """
        list_of_parameters = []
        for module in self.modules:
            if(isinstance(module,Batch_Norm) or isinstance(module,Linear)):
                list_of_parameters.extend(module.give_params())
        return list_of_parameters

    def get_weights(self,lin_module_number):
        """
        Retrieves the weight matrix for a specific linear layer.

        Args:
            lin_module_number: The index of the linear module whose weights are requested.
                              (Corresponds to the count of linear layers).

        Returns:
            np.ndarray: The weight matrix of the specified linear module.
        """
        i = 1
        for module in self.modules:
            if(isinstance(module,Linear)):
                if(i == lin_module_number):
                    return module.weights
                i+=1

    def forward_pass(self,X,training = True):
        """
        Performs the forward pass for the entire neural network.

        Args:
            X: Input data of shape (n_sample_features, batch_size).
            training: Boolean flag indicating whether the network is in training mode or inference mode.

        Returns:
            np.ndarray: The final output of the network (e.g., of shape (10, batch_size) for the MNIST dataset).
        """
        self.X = X
        input = X
        for i in range(len(self.modules)):
            input = self.modules[i].forward(input,training)
        self.net_output = input

    def backward_pass(self,Y):
        """
        Executes the full backpropagation process for the neural network
        and performs optimization steps on all learned parameters.

        Args:
            Y: The ground-truth labels for the input samples.
        """
        #We count dLoss/dNet_output
        grad = backward_loss(self.loss_function,self.net_output,Y)/self.k
        for i in range(self.L-1,-1,-1):
            if(i == self.L-1 and self.loss_function =="NLL"):
                #When our function is NLL we don't count dLoss/dA, because of mathematical simplicity
                #It is much better to count dLoss/dZ, which mean to skip computing gradient for the
                #last activation module in the net
                pass
            else:
                grad = self.modules[i].backward(grad)
        #All the gradient steps are made here
        self.optimizer.step()

    def train(self,X,Y,epoc,k = 32,X_val = None,Y_val = None):
        """
        Trains the neural network using mini-batch gradient descent.

        Args:
            X: Input training data of shape (n_sample_features, batch_size).
            Y: Ground-truth labels for the training data (e.g., shape (10, batch_size) for MNIST).
            epoc: Number of training epochs.
            k: Batch size, default is 32.
            X_val: Optional validation dataset features.
            Y_val: Optional ground-truth labels for the validation dataset.

        Returns:
            None
        """
        data_size = X.shape[1]
        k = data_size if k == None else k
        self.k = k
        for e in range(epoc):
            #We shuffle the data randomly after each epoc
            permutation = np.random.permutation(data_size)
            X_shuffled = X[:, permutation]
            Y_shuffled = Y[:, permutation]
            index = 0
            for index in range(0,data_size,k):
                #We divide dataset for batches
                Batch = X_shuffled[:,index:index+k]
                Batch_Y = Y_shuffled[:,index:index+k]
                self.forward_pass(Batch)
                self.backward_pass(Batch_Y)
            if(self.plot == True):
                #It is necessary to compute accuracy while training if we want to draw a plot
                self.evaluate(X_val,Y_val)
        if(self.plot == True):
            #Here we are drawing the plot
            epochs = [i + 1 for i in range(epoc)]
            draw_accuracy(self.accuracy_history,epochs)


    def predict(self,X):
        """
        Performs a forward pass for the entire neural network to make predictions.

        Args:
            X: Input data to the network of shape (n_sample_features, data_size).

        Returns:
            np.ndarray: The output of the whole network (e.g., class probabilities or logits
                        of shape (10, data_size) for the MNIST dataset).
        """
        self.forward_pass(X,training = False)
        Y_net = self.net_output
        return self.net_output

    def evaluate(self,X_val,Y_val):
        """
        Computes evaluation metrics after the training process using the validation dataset.
        Available metrics include:
            - accuracy: Overall classification accuracy.
            - mistake_matrix: Confusion matrix showing misclassifications (row i, column j represents
              the number of times digit i was misclassified as digit j).
            - per-class_accuracy: Accuracy score for each individual digit class.

        Args:
            X_val: The validation dataset features.
            Y_val: The labels for the validation dataset in a one-hot encoded format.

        Returns:
            dict: A dictionary containing the computed metrics.
        """
        metrics_dictionary = {}
        mistake_matrix = np.zeros((10,10))
        digit_table = np.zeros(10)
        digit_counter = np.zeros(10)
        digit_dictionary = {}
        Y_net = de_code_to_digit(self.predict(X_val))
        Y_val = de_code_to_digit(Y_val)
        mistake = 0
        for i in range(len(Y_val)):
            digit_counter[Y_val[i]] += 1
            if(Y_net[i] != Y_val[i]):
                mistake += 1
                mistake_matrix[Y_net[i]][Y_val[i]] += 1
                digit_table[Y_val[i]] += 1
        self.accuracy_history.append(1-mistake/len(Y_val))
        if("accuracy" in self.metrics):
            metrics_dictionary["accuracy"] = 1-mistake/len(Y_val)
        if("mistake_matrix" in self.metrics):
            metrics_dictionary["mistake_matrix"] = mistake_matrix
        if("per_class_accuracy" in self.metrics):
            digit_table = (digit_counter - digit_table) / digit_counter
            for i in range(10):
                digit_dictionary[str(i)] = round(float(digit_table[i]), 3)
            metrics_dictionary["per_class_accuracy"] = digit_dictionary
        return metrics_dictionary

def main():
    X_train,Y_train,X_val,Y_val = get_MNIST_data()
    print(Y_train.shape,Y_val.shape)
    X_micro = X_train[:,:60000]
    Y_micro = Y_train[:,:60000]
    layers1 = [Linear(784,128),
              Activation(ReLU),
              Linear(128,256),
              Activation(ReLU),
              Linear(256,10),
              Activation(softmax)]
    My_Net = Net(layers1,"NLL",metrics = ["per_class_accuracy","accuracy"],plot = True,lr_rate = 0.001)
    My_Net.train(X_micro,Y_micro,epoc = 10,X_val = X_val,Y_val = Y_val)
    print(My_Net.evaluate(X_val,Y_val))
if __name__ == "__main__":
    main()
