import time 
import numpy as np 

from fully_connected_nn import FullyConnectedNN

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

#load breast cancer dataset
data= load_breast_cancer()

X= data.data
y= data.target.reshape(-1,1)

print("X shape:", X.shape)
print("y shape:", y.shape)

accuracies= []
start_time= time.time()
for i in range(20):
    print(f"\nRun {i+1}")

    #split 70/30
    X_train,X_test,y_train,y_test= train_test_split(X,y,test_size=0.3)

    #normalize data to [0,1]
    scaler= MinMaxScaler()
    X_train= scaler.fit_transform(X_train)
    X_test= scaler.transform(X_test)

    #create model 
    model= FullyConnectedNN(
        input_size=X.shape[1],
        hidden_size=10,
        lr=0.01
    )
    #train model 
    model.fit(X_train,y_train,iterations=2000,batch_size=64,show_step=1000)

    #make predictions 
    y_pred=model.predict(X_test)

    #compute accuracy
    acc= np.mean(y_pred == y_test)
    accuracies.append(acc)

    print("Accuracy:",acc)

mean_acc= np.mean(accuracies)
std_acc= np.std(accuracies)
end_time= time.time()

print("\n---FINAL RESULTS---")
print("Mean Accuracy:", mean_acc)
print("Std:", std_acc)
print("Execution Time:",end_time - start_time,"seconds")
