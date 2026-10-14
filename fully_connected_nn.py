import numpy as np 

class FullyConnectedNN:
    def __init__(self,input_size,hidden_size,output_size=1,lr=1e-2):
        self.input_size = input_size #num of input features
        self.hidden_size = hidden_size #num of hidden neurons
        self.output_size = output_size  #num of output neurons
        self.lr = lr #learning rate

        #weights and biases
        self.W1 = None 
        self.b1 = None
        self.W2 = None 
        self.b2 = None 

        #forward pass variables
        self.Z1 = None 
        self.A1 = None 
        self.Z2 = None
        self.A2 = None 

        #gradients dictionary
        self.gradients = {}

    #ReLU activation function
    def relu(self, Z):
        return np.maximum(0,Z)
   
    #derivative of ReLU
    def relu_derivative(self,Z):
        return (Z > 0).astype(float)

    #sigmoid activation function
    def sigmoid(self, Z):
        return 1/ (1 + np.exp(-Z))
    
     #initialize model parameters
    def init_parameters(self):

        # hidden layer parameters
        self.W1 = np.random.randn(self.hidden_size, self.input_size)*0.1
        self.b1 = np.random.randn(self.hidden_size,1)*0.1
        # output layer parameters
        self.W2 = np.random.randn(self.output_size, self.hidden_size)*0.1
        self.b2 = np.random.randn(self.output_size,1)*0.1
    
    # output layer parameters
    def forward(self, X):
        X = X.T # (features, N)
        
         #hidden layer
        self.Z1= self.W1 @ X + self.b1
        self.A1= self.relu(self.Z1)
        #output layer
        self.Z2= self.W2 @ self.A1 + self.b2
        self.A2= self.sigmoid(self.Z2)
        #return predictions with shape(N,1)
        return self.A2.T
   #predict classes
    def predict(self,X):
        probabilities=self.forward(X)
    
        predictions= (probabilities >= 0.5).astype(int)
        return predictions
    
    #binary cross entropy loss
    def compute_loss(self, y_true, y_pred):
        #make shapes compatible
        y_true = y_true.reshape(y_pred.shape)
        #avoid log(0)
        eps = 1e-8
        y_pred = np.clip(y_pred, eps, 1 - eps)

        loss = -np.mean(
            y_true * np.log(y_pred) +
            (1 - y_true) * np.log(1 - y_pred)
        )

        return loss
    
    def loss(self,X,y):
        y_pred= self.forward(X)
        return self.compute_loss(y, y_pred)
    
    #backpropagation
    def backward(self, X, y):
       m = X.shape[0] #samples
       X_T = X.T #transpose input

       y = y.reshape(1, m) #reshape labels

       dZ2 = self.A2 - y #output layer gradients

       dW2 = (1 / m) * (dZ2 @ self.A1.T)
       db2 = (1 / m) * np.sum(dZ2, axis=1, keepdims=True)

       #hidden layer gradients
       dA1 = self.W2.T @ dZ2
       dZ1 = dA1 * self.relu_derivative(self.Z1)

       dW1 = (1 / m) * (dZ1 @ X_T.T)
       db1 = (1 / m) * np.sum(dZ1, axis=1, keepdims=True)
       #store gradients
       self.gradients = {
         "dW1": dW1,
         "db1": db1,
         "dW2": dW2,
         "db2": db2
        }        
   
   
   #gradient descent update
    def step(self):
        self.W1 -= self.lr*self.gradients["dW1"]
        self.b1 -= self.lr*self.gradients["db1"]

        self.W2 -= self.lr*self.gradients["dW2"]
        self.b2 -= self.lr*self.gradients["db2"]

    #train model
    def fit(self,X,y,iterations=10000,batch_size=None,show_step=1000):
        #check input types
        if not isinstance(X,np.ndarray) or not isinstance(y,np.ndarray):
           raise ValueError("X and y must numpy arrays")
        #check compatible dimensions
        if X.shape[0]!=y.shape[0]:
            raise ValueError("X and y must have same number of samples")

        self.init_parameters()

        N=X.shape[0]

        for i in range(iterations):

            if batch_size is None:
                X_batch = X 
                y_batch = y
            else: 
                idx = np.random.choice(N,batch_size,replace=False)
                X_batch = X[idx]
                y_batch = y[idx]

            #forward propagation
            y_pred= self.forward(X_batch)

            #backward propagation
            self.backward(X_batch,y_batch)

            #update parameters
            self.step()
            
            #print loss 
            if i % show_step == 0:
                current_loss= self.loss(X,y)
                print(f"Iteration {i},Loss:{current_loss}")



