import time 
import numpy as np 
import torch 
import torch.nn as nn 
import torch.optim as optim 

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

data= load_breast_cancer()
X= data.data 
y= data.target.reshape(-1,1)

class NeuralNet(nn.Module):
    def __init__(self, input_size):
        super(NeuralNet, self).__init__()

        self.fc1= nn.Linear(input_size,10)
        self.relu= nn.ReLU()
        self.fc2= nn.Linear(10,1)
        self.sigmoid= nn.Sigmoid()

    def forward(self,x):
        out= self.fc1(x)
        out= self.relu(out)
        out= self.fc2(out)
        out= self.sigmoid(out)
        return out
    
accuracies= []
start_time= time.time()


for i in range(20):
    print(f"\nRun {i+1}")

    #split 
    X_train,X_test,y_train,y_test = train_test_split(
        X,y,test_size=0.3
    )

     #normalize 
    scaler= MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    #Convert data to PyTorch tensors
    X_train= torch.tensor(X_train, dtype=torch.float32)
    y_train= torch.tensor(y_train, dtype=torch.float32)

    X_test= torch.tensor(X_test, dtype=torch.float32)
    y_test= torch.tensor(y_test, dtype=torch.float32)

    model= NeuralNet(X.shape[1])
    criterion= nn.BCELoss()
    optimizer= optim.SGD(model.parameters(),lr=0.01)
        
    for epoch in range(2000):
        outputs= model(X_train)
        loss= criterion(outputs,y_train)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 1000 ==0:
            print(f"Epoch {epoch},Loss:{loss.item()}")
               
    with torch.no_grad():
        y_pred= model(X_test)
        y_pred= (y_pred >= 0.5).float()

        acc= (y_pred == y_test).float().mean().item()
        accuracies.append(acc)
        print("Accuracy:",acc)
        model = NeuralNet(X.shape[1])

mean_acc= np.mean(accuracies)
std_acc= np.std(accuracies)

end_time= time.time()

print("\n---FINAL RESULTS---")
print("Mean Accuracy:",mean_acc)
print("Std:",std_acc)
print("Time:",end_time - start_time)

