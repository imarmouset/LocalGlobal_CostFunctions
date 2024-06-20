import math 
import torch


class SGDOptimizer:
    def __init__(self, params, lr=0.01):
        self.params = params
        self.lr = lr

    def step(self):
        for param in self.params:
            param.data -= self.lr * param.grad.data

    def zero_grad(self):
        for param in self.params:
            param.grad.data.zero_()


class AdamOptimizer:
    def __init__(self, params, lr = 0.01,  beta1=0.9, beta2=0.999, eps=1e-8):
        self.params = params
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = eps
        self.m = [torch.zeros_like(param) for param in params]
        self.v = [torch.zeros_like(param) for param in params]
        self.t = 0
    
    def step(self):
        self.t += 1
        for i, param in enumerate(self.params):
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * param.grad.data
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * param.grad.data ** 2
            m_hat = self.m[i] / (1 - self.beta1 ** self.t)
            v_hat = self.v[i] / (1 - self.beta2 ** self.t)
            param.data -= self.lr * m_hat / (torch.sqrt(v_hat) + self.epsilon)
    
    def zero_grad(self):
        for param in self.params:
            param.grad.data.zero_()
