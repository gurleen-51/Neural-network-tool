import numpy as np
from core.activations import linear, linear_d, sigmoid, sigmoid_d, tanh, tanh_d


# ------------------ INITIALIZATION ------------------
def init_weights(layers):
    weights = [np.random.randn(layers[i], layers[i+1]) * 0.1
               for i in range(len(layers) - 1)]
    biases = [np.zeros((1, layers[i+1]))
              for i in range(len(layers) - 1)]
    return weights, biases


# ------------------ FORWARD PASS ------------------
def forward(X, weights, biases, activation="Sigmoid"):
    a = [X]
    z = []

    act_func = {
        "Sigmoid": (sigmoid, sigmoid_d),
        "Tanh": (tanh, tanh_d),
        "Linear": (linear, linear_d)
    }
    act, act_d = act_func[activation]

    for w, b in zip(weights, biases):
        z_l = a[-1] @ w + b
        z.append(z_l)
        a.append(act(z_l))

    return a, z, act_d


# ------------------ BACKPROP ------------------
def backprop(a, z, y, weights, act_d):
    delta = (a[-1] - y) * act_d(z[-1])
    deltas = [delta]

    for l in range(len(weights) - 2, -1, -1):
        delta = deltas[0] @ weights[l + 1].T * act_d(z[l])
        deltas.insert(0, delta)

    return deltas


# ------------------ UPDATE ------------------
def update_params(weights, biases, a, deltas, lr):
    m = a[0].shape[0]

    for l in range(len(weights)):
        weights[l] -= lr * (a[l].T @ deltas[l]) / m
        biases[l] -= lr * np.mean(deltas[l], axis=0, keepdims=True)

    return weights, biases


# ------------------ TRAIN NETWORK ------------------
def train_network(
    X,
    y,
    layers,
    lr=0.01,
    epochs=100,
    activation="Sigmoid"
):
    # Normalize input (VERY important)
    X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)

    weights, biases = init_weights(layers)
    losses = []
    predictions_per_epoch = []

    for _ in range(epochs):
        a, z, act_d = forward(X, weights, biases, activation)
        loss = np.mean((y - a[-1]) ** 2)
        losses.append(loss)

        predictions_per_epoch.append(a[-1])

        deltas = backprop(a, z, y, weights, act_d)
        weights, biases = update_params(weights, biases, a, deltas, lr)

    return {
        "weights": weights,
        "biases": biases,
        "losses": losses,
        "predictions": predictions_per_epoch,
        "final_output": a[-1]
    }
