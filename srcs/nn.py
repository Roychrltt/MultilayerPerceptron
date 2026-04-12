import numpy as np

class Value:
    def __init__(self, data, _children=(), _op=''):
        self.data = np.array(data, dtype=np.float64)
        self.grad = np.zeros_like(self.data)
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op

    def matmul(self, other):
        out = Value(self.data @ other.data, (self, other), 'matmul')
        def _backward():
            self.grad += out.grad @ other.data.T
            other.grad += self.data.T @ out.grad
        out._backward = _backward
        return out

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), '+')
        def _backward():
            grad_self = out.grad
            grad_other = out.grad
            
            while grad_self.ndim > self.data.ndim:
                grad_self = grad_self.sum(axis=0)
            for i, dim in enumerate(self.data.shape):
                if dim == 1: grad_self = grad_self.sum(axis=i, keepdims=True)
                
            while grad_other.ndim > other.data.ndim:
                grad_other = grad_other.sum(axis=0)
            for i, dim in enumerate(other.data.shape):
                if dim == 1: grad_other = grad_other.sum(axis=i, keepdims=True)

            self.grad += grad_self
            other.grad += grad_other
        out._backward = _backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), '*')
        def _backward():
            grad_self = other.data * out.grad
            grad_other = self.data * out.grad
            
            while grad_self.ndim > self.data.ndim:
                grad_self = grad_self.sum(axis=0)
            for i, dim in enumerate(self.data.shape):
                if dim == 1: grad_self = grad_self.sum(axis=i, keepdims=True)
                
            while grad_other.ndim > other.data.ndim:
                grad_other = grad_other.sum(axis=0)
            for i, dim in enumerate(other.data.shape):
                if dim == 1: grad_other = grad_other.sum(axis=i, keepdims=True)

            self.grad += grad_self
            other.grad += grad_other
        out._backward = _backward
        return out

    def __pow__(self, other):
        assert isinstance(other, (int, float))
        out = Value(self.data**other, (self,), f'**{other}')
        def _backward():
            self.grad += (other * self.data**(other-1)) * out.grad
        out._backward = _backward
        return out

    def relu(self):
        out = Value(np.maximum(0, self.data), (self,), 'ReLU')
        def _backward():
            self.grad += (self.data > 0) * out.grad
        out._backward = _backward
        return out

    def log(self):
        out = Value(np.log(self.data + 1e-15), (self,), 'log')
        def _backward():
            self.grad += (1.0 / (self.data + 1e-15)) * out.grad
        out._backward = _backward
        return out

    def exp(self):
        out = Value(np.exp(self.data), (self,), 'exp')
        def _backward():
            self.grad += out.data * out.grad
        out._backward = _backward
        return out

    def abs(self):
        out = Value(np.abs(self.data), (self,), 'abs')
        def _backward():
            self.grad += np.sign(self.data) * out.grad
        out._backward = _backward
        return out

    def mean(self):
        out = Value(np.mean(self.data), (self,), 'mean')
        def _backward():
            self.grad += (1.0 / self.data.size) * np.ones_like(self.data)
        out._backward = _backward
        return out

    def backward(self):
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)
        
        self.grad = np.ones_like(self.data)
        for v in reversed(topo):
            v._backward()

    def __neg__(self): return self * -1
    def __sub__(self, other): return self + (-other)
    def __radd__(self, other): return self + other
    def __rmul__(self, other): return self * other
    def __truediv__(self, other): return self * other**-1

class Module:
    def zero_grad(self):
        for p in self.parameters():
            p.grad = np.zeros_like(p.data)
    def parameters(self): return []

class Layer(Module):
    def __init__(self, nin, nout, nonlin=True):
        self.w = Value(np.random.randn(nin, nout) * np.sqrt(2./nin))
        self.b = Value(np.zeros((1, nout)))
        self.nonlin = nonlin
    def __call__(self, x):
        act = x.matmul(self.w) + self.b
        return act.relu() if self.nonlin else act
    def parameters(self): return [self.w, self.b]

class MLP(Module):
    def __init__(self, nin, nouts):
        sz = [nin] + nouts
        self.layers = [Layer(sz[i], sz[i+1], nonlin=i!=len(nouts)-1) for i in range(len(nouts))]
    def __call__(self, x):
        for layer in self.layers: x = layer(x)
        return x
    def parameters(self): return [p for layer in self.layers for p in layer.parameters()]
