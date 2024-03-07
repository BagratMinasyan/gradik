import numpy as np
class Node:
    def __init__(self, x=0, child=[], parent=None):
        self.x = x
        self.child = child
        self.parent = parent
        self.grad = 0
        self.op = None

    def __repr__(self, ):
        return f'Node(x={self.x})'

    def __add__(self, other):
        other = other if isinstance(other, Node) else Node(other)
        node = Node(self.x + other.x, child=[self, other])
        node.op = '+'
        return node

    def __mul__(self, other):
        other = other if isinstance(other, Node) else Node(other)
        node = Node(self.x * other.x, child=[self, other])
        node.op = '*'
        return node

    def exp(self, ):
        node = Node(np.exp(self.x), child=[self])
        node.op = 'exp'
        return node

    def sigmoid(self, ):
        node = Node(1 / (1 + np.exp(-self.x)), child=[self])
        node.op = 'sigmoid'
        return node

    def loss(self, other):
        node = Node((self.x - other.x) ** 2, child=[self, other])
        node.op = 'l2'
        return node

    def loss(self, other, tipo):
        if tipo == 'ce':
            if self.x == 0:
                self.x = 10 ** -100
            if self.x == 1:
                self.x = 0.999
            ce = np.array([np.log(self.x), np.log(1 - self.x)])
            node = Node(-other.x @ ce, child=[self])
            node.op = 'ce'
        elif tipo == 'l2':
            node = Node((self.x - other.x) ** 2, child=[self, other])
            node.op = 'l2'
        return node

    def back(self):
        if self.op == '+':
            self.child[0].grad += self.grad
            self.child[1].grad += self.grad
        elif self.op == '*':
            self.child[0].grad += self.grad * self.child[1].x
            self.child[1].grad += self.grad * self.child[0].x
        elif self.op == 'exp':
            self.child[0].grad += self.grad * self.x
        elif self.op == 'sigmoid':
            self.child[0].grad += self.grad * (self.x) * (1 - self.x)
        elif self.op == 'l2':
            self.child[0].grad += self.grad * 2 * (self.child[0].x - self.child[1].x)
        elif self.op == 'ce':
            if self.x == 0:
                self.x = 10 ** -100
            self.child[0].grad += self.grad / self.x

    def backward(self, ):
        self.grad = 1
        topo = []
        visited = set()

        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v.child:
                    build_topo(child)
                topo.append(v)

        build_topo(self)
        for node in reversed(topo):
            node.back()
        return topo