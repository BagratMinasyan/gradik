import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import r2_score, confusion_matrix
from sklearn.model_selection import train_test_split
from Model import Model

x,y=load_breast_cancer(return_X_y=True)
x_train,x_test,y_train,y_test=train_test_split(x,y,train_size=0.7)
scaler=StandardScaler().fit(x_train)
x_train=scaler.transform(x_train)
x_test=scaler.transform(x_test)

model=Model(arch=[[x.shape[1],1]],act=['sigmoid'],loss='ce',alfa=0.2)
model.fit(x_train,y_train,epoch=20,batch_size=4)

y_pred=model.pred(x_test)<0.5
print(confusion_matrix(y_test,y_pred))

model=Model(arch=[[x.shape[1],1]],act=['sigmoid'],loss='l2',alfa=0.2)
model.fit(x_train,y_train,epoch=20,batch_size=4)

y_pred=model.pred(x_test)>0.5
print(confusion_matrix(y_test,y_pred))

