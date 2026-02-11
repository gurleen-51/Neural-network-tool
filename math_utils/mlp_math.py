import numpy as np

def mlp_math_steps(X, y, weights, biases, activation):
    """
    Returns a list of step-by-step calculations for first sample in MLP.
    X: input sample(s)
    y: target sample(s)
    weights, biases: list of layers
    activation: 'Sigmoid', 'Tanh', 'Linear'
    """
    steps = []
    for i in range(X.shape[0]):
        steps.append(f"Sample {i+1}: {X[i]}")
        a = X[i:i+1]
        for l in range(len(weights)):
            z = a @ weights[l] + biases[l]
            steps.append(f"Layer {l+1} linear combination z: {z}")
            if activation == "Sigmoid":
                a = 1 / (1 + np.exp(-z))
            elif activation == "Tanh":
                a = np.tanh(z)
            else:
                a = z
            steps.append(f"Layer {l+1} activation a: {a}")
        steps.append(f"Target (y): {y[i]}")
        steps.append("---")
    return steps
