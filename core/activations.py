import numpy as np

def linear(z): return z
def linear_d(z): return np.ones_like(z)

def sigmoid(z): return 1 / (1 + np.exp(-z))
def sigmoid_d(z): return sigmoid(z) * (1 - sigmoid(z))

def tanh(z): return np.tanh(z)
def tanh_d(z): return 1 - np.tanh(z)**2
