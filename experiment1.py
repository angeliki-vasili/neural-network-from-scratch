import numpy as np 
from fully_connected_nn import FullyConnectedNN
from generate_datasets import generate_binary_problem
from helpers import plot_decision_boundary

#data generation
C=np.array([[0,8],
            [0,8]])

X,y = generate_binary_problem(C,N=500)

#split 70/30
split = int(0.7* X.shape[0])
X_train,X_test= X[:split],X[split:]
y_train,y_test= y[:split], y[split:]

#reshape labels
y_train= y_train.reshape(-1,1)
y_test= y_test.reshape(-1,1)

#model generation
model= FullyConnectedNN(input_size=2,hidden_size=3,lr=0.01)

#training
model.fit(X_train,y_train,iterations=5000,show_step=500)

#predictions 
y_pred= model.predict(X_test)

#accuracy
accuracy= np.mean(y_pred == y_test)
print("Accuracy:",accuracy)

#plot decision boundary
plot_decision_boundary(model.predict,X,y)
print("\n ---SECOND DATASET---")
C2=np.array([[0,2],
             [0,2]])

X2,y2 =generate_binary_problem(C2,N=500)

#spilt
spilt=int(0.7*X2.shape[0])
X_train2,X_test2=X2[:spilt],X2[spilt:]
y_train2,y_test2=y2[:split],y2[spilt:]

y_train2= y_train2.reshape(-1,1)
y_test2= y_test2.reshape(-1,1)

#new model 
model2=FullyConnectedNN(input_size=2,hidden_size=3,lr=0.01)

#training 
model2.fit(X_train2,y_train2,iterations=5000,show_step=500)

#accuracy 
y_pred2= model2.predict(X_test2)
accuracy2= np.mean(y_pred2==y_test2)

print("Accuracy (harder):",accuracy2)

#plot 
plot_decision_boundary(model2.predict,X2,y2)

print("\n ---FLOWER PROBLEM---")
from generate_datasets import generate_flower_problem
#generate data
Xf,yf =generate_flower_problem()
Xf= Xf.T
yf= yf.reshape(-1,1)

#split into train/test(70/30)
split = int(0.7* Xf.shape[0])
X_train_f,X_test_f= Xf[:spilt],Xf[spilt:]
y_train_f,y_test_f= yf[:spilt],yf[spilt:]


#create model 
model_f= FullyConnectedNN(input_size=2, hidden_size=3, lr=0.01)

#train model 
model_f.fit(X_train_f,y_train_f,iterations=5000,show_step=500)

#make predictions
y_pred_f= model_f.predict(X_test_f)

#compute accuracy
acc_f = np.mean(y_pred_f == y_test_f)
print("Accuracy (flower):", acc_f)
plot_decision_boundary(model_f.predict,Xf,yf.ravel())

