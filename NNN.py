import numpy as np # helps with the math
import matplotlib.pyplot as plt # to plot error during training
import NN_Visualiazation_matplotlib as vis
global x,y 
x = 0
y = 0
# input data
inputs = np.array([[1, 1, 0],
                   [0, 1, 1],
                   [1, 0, 1],
                   [0, 1, 0],
                   [0, 1, 0],
                   ])
# output data
outputs = np.array([[1], [0], [1], [0], [0]])
'''
inputs = np.array([[1, 1, 0]])
outputs = np.array([[1]])'''


# create NeuralNetwork class
class NeuralNetwork:

    # intialize variables in class
    def __init__(self, inputs, outputs):
        self.inputs  = inputs
        self.outputs = outputs
        # initialize weights as .50 for simplicity
        self.weights = np.array([[.50], [.50], [.50]])
        self.error_history = []
        self.epoch_list = []

    #activation function ==> S(x) = 1/1+e^(-x)
    def sigmoid(self, x, deriv=False):
        if deriv == True:
            return x * (1 - x)
        return 1 / (1 + np.exp(-x))

    # data will flow through the neural network.
    def feed_forward(self):
        global x
        x += 1
        self.hidden = self.sigmoid(np.dot(self.inputs, self.weights))
        #if x == 100000:
            #print(f'inputs: {self.inputs}, \n \n weights: {self.weights}, \n \n net: {np.dot(self.inputs, self.weights)}, \n \n sig: {self.hidden}')

    # going backwards through the network to update weights
    def backpropagation(self):
        global y
        y += 1
        self.error  = self.outputs - self.hidden
        delta = self.error * self.sigmoid(self.hidden, deriv=True)
        #if y == 10000:
            #print(f'delta: {delta}, \n \n factor: {self.sigmoid(self.hidden, deriv=True)}, \n \n weights: {self.weights}\n \n grad: {np.dot(self.inputs.T, delta)} \n \n unknown:{self.inputs.T}')
        self.weights += np.dot(self.inputs.T, delta)
        if y == 10000:
            print(f'weightnew:\n {self.weights}')
        example_2 = np.array([[1, 0, 0]])
        error = abs(NN.predict(example_2)-1)
        #f = open('example1.txt', "a")
        #f.write(f"{y};{str(self.weights[0])[1:-1]};{str(self.weights[1])[1:-1]};{str(self.weights[2])[1:-1]};{str(error)[2:-2]}\n")
    # train the neural net for 25,000 iterations
    def train(self, epochs=10000):
        template = []
        for epoch in range(epochs):
            # flow forward and produce an output
            self.feed_forward()
            # go back though the network to make corrections based on the output
            self.backpropagation()    
            # keep track of the error history over each epoch
            self.error_history.append(np.average(np.abs(self.error)))
            self.epoch_list.append(epoch) 
            # template = [[input1,input2,input3],[weights1, weights2, weights3],[nodes],error,[nodevalue1,nodevalue2,nodevalue3], output value,n]
            val = NN.predict([1,0,1]) 
            error = 1 - val
            a = [1,0,0]
            b = [float(self.weights[0]),float(self.weights[1]),float(self.weights[2])]
            bc = [round(float(self.weights[0]+self.weights[2]),3)]
            c = round(float(error),6)
            d = a
            e = round(float(val[-1]),6)
            f = epoch+1
            template.append([a,b,bc,c,d,e,f])
        return(template)

    # function to predict output on new and unseen input data                               
    def predict(self, new_input):
        prediction = self.sigmoid(np.dot(new_input, self.weights))
        return prediction

                    
# create neural network   
NN = NeuralNetwork(inputs, outputs)
# train neural network
out = (NN.train())
for i in range(len(out)):
    print(str(i)+'. round sucessfully done!')
    #print(out[i])
    vis.i(out[i])

