import numpy as np

class Network():
    def __init__(self, input_size, output_size, hidden_layers):
        # Build a neural network with 
        # input size: 'input_size'
        # output size: 'output_size'
        # number of hidden layers as 'hidden_layers' where
        # hidden_layers is a list of number nodes in layer
        
        ############### Write code ######################
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_layers = hidden_layers




        # Define and initialize Weights and Bias in Dictionary format
        init_coef = 0.1
        self.num_layers = len(hidden_layers)+1 # 4
        self.num_hidden_layers = len(hidden_layers) #3
        self.W = {}
        self.b = {}
        # 첫번째 layer weight bias 초기화
        self.W[1] = init_coef * np.random.randn(self.input_size, self.hidden_layers[0]) #input과 h1
        self.b[1] = init_coef * np.random.randn(self.hidden_layers[0])

        # Initialize rest of the weights and bias in store in self.W and self.b
        for i in range(self.num_hidden_layers - 1):#h1,h2 ~ h[num_layers-2],h[num_layers-1]
            self.b[i+2] = init_coef * np.random.randn(self.hidden_layers[i+1])
            self.W[i+2] = init_coef * np.random.randn(self.hidden_layers[i], self.hidden_layers[i+1])
        # 마지막 layer weight 초기화
        self.b[self.num_layers] = init_coef * np.random.randn(self.output_size)
        self.W[self.num_layers] = init_coef * np.random.randn(self.hidden_layers[self.num_hidden_layers - 1], self.output_size)

        


        
        
    def forward(self, x):
        
        self.Z = {}
        self.A = {}
        ############### Write code ######################
        # | w1, b1  | w2 b2 | w3 b3 | w4 b4 |
        self.Z[1] = np.matmul(x, self.W[1]) + self.b[1]
        
        for i in range(1,self.num_hidden_layers + 1):
            self.A[i] = ReLU(self.Z[i])
            self.Z[i+1] = np.matmul(self.A[i], self.W[i+1]) + self.b[i+1]
        
        self.O = softmax(self.Z[self.num_layers])
        return self.O
       
    
    def backward(self, X, y):
        self.delta = {}
        self.dW = {}
        self.db = {}

        self.Y = one_hot(y, self.output_size)
        self.e = crossentropy(self.O, self.Y)
        self.loss = np.sum(self.e) / X.shape[0]
        self.delta[self.num_layers] = self.O - self.Y #cross-entropy와 softmax 미분값

        for i in reversed(range(2, self.num_layers + 1)):
            # self.db[i] = np.sum(self.delta[i], axis=0)
            self.db[i] = np.matmul(np.ones((1, X.shape[0])), self.delta[i]) / X.shape[0]
            print(np.ones(X.shape[0]).shape, self.delta[i].shape)
            self.dW[i] = np.matmul(np.transpose(self.A[i-1]), self.delta[i]) / X.shape[0] #외적계산이긴한데 배치를 묶어주어서 전미분으로 계산을 하면 내적계산으로 변한다.
            self.delta[i-1] = np.matmul(self.delta[i], np.transpose(self.W[i])) *  dReLU(self.Z[i-1])

        self.db[1] = np.matmul(np.ones((1, X.shape[0])), self.delta[1]) / X.shape[0]
        self.dW[1] = np.matmul(X.T, self.delta[1]) / X.shape[0]
        
    def SGD(self):
        learning_rate = 0.1
        for i in range(1, self.num_layers):
            self.W[i] = self.W[i] - learning_rate * self.dW[i]
            self.b[i] = self.b[i] - learning_rate * self.db[i]
        

        
def ReLU(x):
    return np.fmax(0, x)

def dReLU(x):
    return np.maximum(0, 1)

def softmax(z):
    O = []
    for i in range(z.shape[0]):
        O.append(np.exp(z[i]) / np.sum(np.exp(z[i])))
    O = np.vstack(O)
    return O


        
def one_hot(y, length_of_onehot):
    ########## code goes here
    ls = []
    for i in range(len(y)):
        ls.append(np.eye(length_of_onehot)[y[i]])
    return np.vstack(ls)
        
        
def crossentropy(O, Y):
    
    ########## code goes here
    return np.sum(-np.log(O + 0.0001) * Y, axis=1)
