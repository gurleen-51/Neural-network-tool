import numpy as np

def perceptron_math_steps(X, y, weights, biases, activation):
    """
    Returns a list of step-by-step calculations for perceptron.
    X: input sample(s)
    y: target sample(s)
    weights, biases: trained parameters
    activation: 'Sigmoid', 'Tanh', 'Linear'
    """
    steps = []
    for i in range(X.shape[0]):
        steps.append(f"Sample {i+1}: {X[i]}")
        z = X[i] @ weights[0] + biases[0]
        steps.append(f"Linear combination (z): {z}")
        if activation == "Sigmoid":
            a = 1 / (1 + np.exp(-z))
        elif activation == "Tanh":
            a = np.tanh(z)
        else:
            a = z
        steps.append(f"Activation output (a): {a}")
        steps.append(f"Target (y): {y[i]}")
        steps.append("---")
    return steps
