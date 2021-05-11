import numpy as np
import matplotlib.pyplot as plt
import csv
import pandas as pd
from datetime import date
import random

class ANN():
    '''
    Two layer neural network
    '''
    def __init__(self, layers=[8,5,1], learning_rate=0.1, iterations=500, momentum=0.9, activation='sigmoid', learning_function = 'annealing'):
        print("Predictors:", layers[0])
        print("Hidden nodes:", layers[1])
        print("Output nodes:", layers[2])
        print("Learning rate:", learning_rate)
        print("Iterations:", iterations)
        print("Momentum value:", momentum)
        self.params = {}
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.layers = layers
        self.momentum = momentum
        self.trainRMSE = []
        self.trainCE = []
        self.validateRMSE = []
        self.validateCE = []
        self.initialLearn = learning_rate
        #User can choose a function for their learning_rate
        if learning_function == 'bold_driver':
            self.learning_function = self.bold_driver
            self.learning = 'Bold Driver'
            print("Learning function: Bold Driver")
        elif learning_function == 'annealing':
            self.learning_function = self.annealing
            self.learning = 'Annealing'
            print("Learning function: Annealing")
        else:
            self.learning_function = 'none'
            self.learning = 'None'
            print("No learning function selected")
        #User can choose activation function to be sigmoid or tanh
        if activation == 'sigmoid':
            self.activation = self.sigmoid
            self.gradActivation = self.gradientSigmoid
            self.active = 'Sigmoid'
            print("Activation function: Sigmoid")
        elif activation == 'tanh':
            self.activation = self.tanh
            self.gradActivation = self.gradientTanh
            self.active = 'Tanh'
            print("Activation function: TanH")

    def init_weights(self):
        '''
        Randomly initialize the weights from normal distribution
        '''
        #layer[0] = no. of input nodes, layer[1] = no. of hidden nodes, layer[2] = no. of output nodes
        #Set low values and high values to [-2/n, 2/n] with n = no. of input nodes
        low = (-2 / self.layers[0])
        high = (2 / self.layers[0])
        #Using numpys random generator create numpy arrays of relevant shape
        #W1 - array of input weights for hidden layer, shape is no. of input nodes x no. of hidden nodes
        self.params['W1'] = np.random.uniform(low, high, size = (self.layers[0], self.layers[1]))
        #b1 - array of biases for hidden nodes, shape is no. of hidden nodes x 1
        self.params['b1'] = np.random.uniform(low, high, size = (self.layers[1],))
        #W2 - array of output weights for hidden layer, shape is no. of hidden nodes x no. of output nodes
        self.params['W2'] = np.random.uniform(low, high, size = (self.layers[1],self.layers[2]))
        #b2 - array of biases for output nodes, shape is no. of output nodes x 1
        self.params['b2'] = np.random.uniform(low, high, size = (self.layers[2],))

        #For use with momentum, initialise arrays of same sizes as their corresponding arrays, but initially set to 0
        #Used to log the change of each weight/bias during previous change of weights/biases
        self.params['W1change'] = np.zeros((self.layers[0], self.layers[1]))
        self.params['b1change'] = np.zeros((self.layers[1],))
        self.params['W2change'] = np.zeros((self.layers[1],self.layers[2]))
        self.params['b2change'] = np.zeros((self.layers[2],))

    def sigmoid(self, Z):
        '''
        Used to model input as value between 0 and 1
        '''
        return (1 / (1 + np.exp(-Z)))

    def gradientSigmoid(self, Z):
        '''
        Used in back_propogation where Z is value of sigmoid function
        '''
        return (Z * (1 - Z))
        
    def tanh(self, Z):
        '''
        Used as alternative to sigmoid function
        '''
        return np.tanh(Z)

    def gradientTanh(self, Z):
        '''
        Used as alternative to gradient sigmoid function
        '''
        return 1.0 - Z**2

    def forward_pass(self, X, Y):
        #Compute weighted sum between input and first layer's weights, add bias
        Z1 = X.dot(self.params['W1']) + self.params['b1']
        #Pass result through activation function
        A1 = self.activation(Z1)
        #Compute weighted sum of output (A1) and second layer's weights, add bias
        Z2 = A1.dot(self.params['W2']) + self.params['b2']
        #Pass result through activation function
        y1 = self.activation(Z2)

        #Calculate error of the output node
        #Observed value - modelled value * gradient activation of modelled value
        delta_output = (Y - y1) * self.gradActivation(y1)
        #Save weighted sum of hidden nodes for use in backprop
        self.params['A1'] = A1

        return y1, delta_output

    def back_propagation(self, delta_output, X):
        '''
        Back propogation algorithm used for backard pass
        '''
        
        #Calculate error for each hidden node
        #Multiply delta_output by output weight of node, then multiply by gradient activation of weighted sum of node
        delta_hidden = delta_output.dot(self.params['W2'].T) * self.gradActivation(self.params['A1'])

        #Update weights and biases with MOMENTUM
        #For W1 (weights going into hidden nodes) calculate change in weight by multipling learning rate by delta of the hidden node
        #it corresponds to and the value of the input node the weight corresponds to (X)
        #then add the momentum value * the previous weight change
        self.params['W1change'] = self.learning_rate * delta_hidden * X[:, None] + (self.momentum * self.params['W1change'])
        #For W2 (weights going out of hidden nodes) calculate change in weight by multiplying learning rate by delta of output node
        #then multiply by weighted sum of corresponding hidden node (A1) and add momentum * previous weight change
        self.params['W2change'] = self.learning_rate * delta_output * self.params['A1'][:, None] + (self.momentum * self.params['W2change'])
        #For b1 and b2 calculate change by: learning rate * delta of hidden/output node (respectively) + momentum * previous change
        self.params['b1change'] = self.learning_rate * delta_hidden + (self.momentum * self.params['b1change'])
        self.params['b2change'] = self.learning_rate * delta_output + (self.momentum * self.params['b2change'])
        #Add the change in weights calculated above to current weight / bias arrays
        self.params['W1'] += self.params['W1change']
        self.params['W2'] += self.params['W2change']
        self.params['b1'] += self.params['b1change']
        self.params['b2'] += self.params['b2change']

        '''
        #Update weights and biases without momentum
        self.params['W1'] = self.params['W1'] + self.learning_rate * delta_hidden * X[:, None]
        self.params['W2'] = self.params['W2'] + self.learning_rate * delta_output * self.params['A1'][:, None]
        self.params['b1'] = (self.params['b1'] + self.learning_rate * delta_hidden).flatten()
        self.params['b2'] = self.params['b2'] + self.learning_rate * delta_output
        '''   
        
    def train(self, xTrain, yTrain, xValidate, yValidate):
        '''
        Trains the neural network using the supplied data
        '''
        self.init_weights() #initialize weights and bias
        
        self.trainMean = np.sum(yTrain) / len(yTrain) #Calculate mean of training data
        self.validateMean = np.sum(yValidate) / len(yValidate) #Calculate mean of validation data
 
        for epoch in range(1, self.iterations + 1): #loop through data self.iterations times
            
            '''
            Can be used to stop training when validation error increases
            if len(self.validateCE) > 3 and self.validateCE[-1] < self.validateCE[-3]:
                break
            '''
            errors = [] #initialise errors and meanErrors arrays to be empty at start of each epoch
            meanErrors = []
            #Go through data line by line, perform forward pass and back prop
            for j in range(len(xTrain)):
                #Return modelled output (y1) and delta of output node in forward pass
                y1, delta_output = self.forward_pass(xTrain[j], yTrain[j])
                #Perform back propogation for line of data
                self.back_propagation(delta_output, xTrain[j])
                #Add value of (modelled output - observed output)^2 to error array
                #Used to calculate RMSE and Coefficient of Efficiency
                errors.append((y1 - yTrain[j]) ** 2)
                #Add value of (observed value - mean of observed values)^2 to meanErrors
                #Used to calculate Coefficient of Efficiency
                meanErrors.append((yTrain[j] - self.trainMean) ** 2)

                #Calculate RMSE and Coefficient of Efficiency and store in array
            self.trainRMSE.append(np.sqrt(np.sum(errors) / len(errors)))
            self.trainCE.append(1 - (np.sum(errors) / np.sum(meanErrors)))

            #Run forward passes with validation data and compute errors
            self.test(xValidate, yValidate)
            
            if epoch % 50 == 0:
                print("Epoch numer: ", epoch)
            
            if self.learning_function == self.bold_driver:
                if epoch % 500 == 0:
                    self.bold_driver(4, 10)
            elif self.learning_function == self.annealing:
                self.learning_rate = self.annealing(0.1, 0.01, epoch)

        self.save_network(epoch)
        self.plot_ce()
        self.plot_rmse()
        
    def test(self, xTest, yTest):
        '''
        Validate the MLP on unseen data
        '''
        errors = [] #Initialise error and meanError arrays
        meanErrors = []
        #Cycle through data line by line performing forward pass
        for j in range(len(xTest)):

            y1, delta_output = self.forward_pass(xTest[j], yTest[j])
            #Calculate error values for RMSE and coefficent of efficiency
            errors.append((y1 - yTest[j]) ** 2)
            meanErrors.append((yTest[j] - self.validateMean) ** 2)
        #Calculate RMSE and Coefficienct of Effiency and add to list
        self.validateRMSE.append(np.sqrt(np.sum(errors) / len(errors)))
        self.validateCE.append(1 - (np.sum(errors) / np.sum(meanErrors)))

    def bold_driver(self, percent, frequency):
        '''
        Readjusts value for learning_rate every 1000 epochs if error has fluctuated by > 4%
        Takes arguments: percent - variation in error function that invokes change
                         frequency - how frequently bold_driver is run, so can get old rmse
        '''
        print("old learning rate:",self.learning_rate)
        #Calculate % error change ((current rmse - old rmse) / old rmse) * 100
        errorChange = ((float(self.trainRMSE[-1])-self.trainRMSE[-frequency])/self.trainRMSE[-frequency])*100
        print("error change:", errorChange)
        #if > 4% reduction in error then increase learning rate by 5%, never more than 0.5
        if errorChange < -percent:
            if self.learning_rate * 1.05 > 0.5:
                self.learning_rate = 0.5
            else:
                self.learning_rate *= 1.05
        elif errorChange > percent:
            #if > 4% increase in error then decrease learning paramter by 30%, but never less than 0.01
            if self.learning_rate * 0.7 < 0.01:
                self.learning_rate = 0.01
            else:
                self.learning_rate *= 0.7
            #revert back to weights/biases saved when bold_driver last called
            self.params['W1'] = self.params['W1save']
            self.params['W2'] = self.params['W2save']
            self.params['b1'] = self.params['b1save']
            self.params['b2'] = self.params['b2save']
        #Save current state of weights/biases in case need to revert
        self.params['W1save'] = self.params['W1']
        self.params['W2save'] = self.params['W2']
        self.params['b1save'] = self.params['b1']
        self.params['b2save'] = self.params['b2']

        print("new learning rate:", self.learning_rate)

    def annealing(self, start, end, epoch):
        '''
        Reduce the learning paramter as training progresses
        '''
        return end + (start - end) * (1 - (1 / (1 + np.exp(10 - ((20 * epoch) / self.iterations)))))

    def weight_decay(self, Y, y1, epoch):
        '''
        Penalise large weights, return new error value for delta_output calculation
        '''
        sumWeights = np.sum(self.params['W1'] ** 2) + np.sum(self.params['W2'] ** 2) +\
                     np.sum(self.params['b1'] ** 2) + np.sum(self.params['b2'] ** 2)
        omega = 0.5 * (sumWeights)
        regParam = 1 / (self.learning_rate * epoch)
        return (Y - y1) + omega * regParam
            
    def plot_ce(self):
        '''
        Plots the loss curve
        '''
        plt.plot(range(len(self.trainCE)), self.trainCE, label = "Training")
        plt.plot(range(len(self.validateCE)), self.validateCE, label = "Validation")
        plt.xlabel('Epochs')
        plt.title('Graph of coefficient of efficiency')
        plt.legend(loc='lower right')
        plt.savefig('Networks/{:.5f}CE.png'.format(float(self.validateCE[-1])))
        plt.close()
        plt.clf()

    def plot_rmse(self):
        plt.plot(range(len(self.trainRMSE)), self.trainRMSE, label = "Training")
        plt.plot(range(len(self.validateRMSE)), self.validateRMSE, label = "Validation")
        plt.xlabel('Epochs')
        plt.ylabel('RMSE')
        plt.title('Graph for RMSE')
        plt.legend(loc='upper right')
        plt.savefig('Networks/{:.5f}RMSE.png'.format(float(self.validateCE[-1])))
        plt.close()
        plt.clf()


    def save_network(self, epoch):
        '''
        Saves the current state of network to a text file
        '''
        #Save name of file as validation coefficient of efficiency and date
        today = date.today()
        name = float(self.validateCE[-1])
        currentDate = today.strftime("%d-%m")
        information = [self.learning_rate, epoch, self.momentum, self.activation, self.learning_function]
        #Save layers (number of nodes in each layer), and all weights / biases including extra info about the network
        with open("Networks/{:.5} {}.txt".format(name, currentDate), "ab") as f:
            np.savetxt(f, self.layers, fmt='%i')
            np.savetxt(f, self.params['W1'], delimiter = ',')
            np.savetxt(f, self.params['W2'], delimiter = ',')
            np.savetxt(f, self.params['b1'])
            np.savetxt(f, self.params['b2'])
            np.savetxt(f, information, delimiter=" ", fmt = "%s")
        #Append .csv file of saved MLPs with information about current MLP
        df_table = pd.DataFrame({'Hidden Nodes':self.layers[1], 'Coefficient of efficiency':name, 'RMSE':self.validateRMSE[-1],\
         'Initial Learning Rate':self.initialLearn, 'Momentum':self.momentum, 'Epochs':epoch, 'Activation Function':self.active,\
         'Learning-rate Function':self.learning}, index=[0])
        df_table.to_csv("Networks.csv", mode='a', index=False, header=None)

        print("Efficiency:",name)
        print("Network saved successfully\n\n---\n\n")

def standardise(df):
    '''
    Standardises input dataframes between 0.1-0.9
    '''
    for column in df.columns: 
        df[column] = 0.8 * (df[column] - df[column].min()) / (df[column].max() - df[column].min()) + 0.1
    return df

def loadNetwork(fileName):
    '''
    Loads a saved network, including layers, and states of weights / biases
    '''
    print("LOADING NEURAL NETWORK...")
    #Obtain layers values from first three rows of file
    layers = np.loadtxt(fileName, dtype=int, delimiter = ',', max_rows=3)
    nn = ANN(layers = layers)
    #Variables to indicate start lines of each set of data
    W2_start = layers[0] + 3
    b1_start = W2_start + layers[1]
    b2_start = b1_start + layers[1]
    #Set weights and variables based on data in the file
    nn.params['W1'] = np.loadtxt(fileName, delimiter = ',', skiprows=3, max_rows=layers[0])
    nn.params['W2'] = np.loadtxt(fileName, delimiter = '\n', skiprows=W2_start, max_rows=layers[1]).reshape(layers[1],1)
    nn.params['b1'] = np.loadtxt(fileName, delimiter = '\n', skiprows=b1_start, max_rows=layers[1])
    nn.params['b2'] = np.loadtxt(fileName, delimiter = '\n', skiprows=b2_start, max_rows=1).reshape(1)
    print("Successfully loaded network :)")

    return nn


if __name__ == "__main__":      

    # add header names
    headers = ['AREA','BFIHOST','FARL','FPEXT','LDP','PROPWET','RMED-1D','SAAR','Index flood']
    df = pd.read_csv('CleanData.csv', names=headers)
    print("Unstandardised: \n", df.head())

    dfTest = pd.read_csv('CleanDataTest.csv', names=headers)

    #apply normalization techniques 
    df = standardise(df)
    dfTest = standardise(dfTest)

    #Split data so 60% is training data
    train = df.sample(frac=0.75)
    #Drop all training data from the validation data
    validate = df.drop(train.index)
    #Split data into predictors and predictand
    xTrain = np.asarray(train.drop(columns=['Index flood']))
    yTrain = np.asarray(train.filter(['Index flood'])).flatten()

    xValidate = np.asarray(validate.drop(columns=['Index flood']))
    yValidate = np.asarray(validate.filter(['Index flood'])).flatten()

    xTest = np.asarray(dfTest.drop(columns=['Index flood']))
    yTest = np.asarray(dfTest.filter(['Index flood'])).flatten()
    

    #Generate 500 networks with different characteristics and train them
    for i in range(500):
        layers = [8, np.random.randint(4, 17), 1]
        learning_rate = np.random.randint(1, 50) / 100
        iterations = np.random.randint(500,15000)
        momentum = np.random.randint(5,9) / 10
        activation = random.choice(['sigmoid', 'tanh'])
        learning_function = random.choice(['bold_driver', 'annealing', 'none'])

        nn = ANN(layers, learning_rate, iterations, momentum, activation, learning_function)
        nn.train(xTrain, yTrain, xValidate, yValidate)
    
    '''
    #Used to load and test a network against test data
    nn = loadNetwork("Networks/0.89219 22-03.txt") 
    nn.test(xTest, yTest)
    '''
    