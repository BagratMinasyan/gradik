from Functional import *
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils import shuffle
import numpy as np

class Model:
    def __init__(self, arch=None, act=['sigmoid'], loss='l2', alfa=0.01):
        self.alfa = alfa
        self.loss = loss
        self.arch = []
        for i in range(len(arch)):
            self.arch.append(Linear(arch[i][0], arch[i][1]))
            if act[i] == 'sigmoid':
                self.arch.append(Sigmoid())
            elif act[i] == 'relu':
                pass
            else:
                pass

    def fit(self, x, y, epoch=1, batch_size=1):
        if self.loss == 'ce':
            y = OneHotEncoder(sparse_output=False).fit_transform(y.reshape(-1, 1))
        params = self.params()
        for j in range(epoch):
            batch = 0
            for i in range(len(x)):
                _x = x[i]
                for layer in self.arch:
                    _x = layer(_x)
                _y = Node(y[i])
                loss = _x.loss(_y, self.loss)
                loss.backward()
                batch += 1
                if batch == batch_size:
                    for node in params:
                        node.x -= self.alfa * node.grad / batch_size
                        node.grad = 0
                    batch = 0
            x, y = shuffle(x, y)

    def pred(self, x):
        a = []
        for i in range(len(x)):
            _x = x[i]
            for layer in self.arch:
                _x = layer(_x)
            a.append(_x.x)
        return np.array(a)

    def params(self, ):
        p = []
        for i in self.arch:
            p.extend(i.params())
        return p