import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_diabetes
from sklearn.metrics import r2_score, confusion_matrix
from sklearn.model_selection import train_test_split
from Model import Model

x,y=load_diabetes(return_X_y=True)
model=Model(arch=[[10,1]],act=[''],loss='l2',alfa=0.1)
model.fit(x,y,epoch=100,batch_size=8)
y_pred=model.pred(x)
print(r2_score(y,y_pred))

from sklearn.linear_model import LinearRegression
model1=LinearRegression()
model1.fit(x,y)
y_pred1=model1.predict(x)
print(r2_score(y,y_pred1))

plt.scatter(y_pred,y_pred1, c='g',linewidth=2, label='model pred and sklearn pred')
plt.plot(y,y, c='r',linewidth=4,label='ground truth')
plt.legend()
plt.show()