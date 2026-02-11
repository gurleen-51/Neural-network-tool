import streamlit as st
import pandas as pd
import numpy as np
from core.network import train_network
from visuals.plots import plot_loss
from visuals.diagrams import network_diagram
from math_utils.mlp_math import mlp_math_steps

def playground_section():
    st.markdown("## 🏗 Playground - Build & Train Neural Networks")
    st.markdown("Upload your dataset or use a random one to start experimenting!")

    # ================== Dataset Selection ==================
    use_csv = st.checkbox("📂 Use CSV Dataset")
    if use_csv:
        uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            df = df.select_dtypes(include=[np.number])
            st.markdown("### Dataset Preview")
            st.dataframe(df.head())

            X = df.iloc[:, :-1].values
            y = df.iloc[:, -1].values.reshape(-1, 1)
        else:
            st.warning("Upload a CSV to proceed!")
            return
    else:
        n_input = st.number_input("Input Features", 1, 10, 3)
        n_output = st.number_input("Output Features", 1, 5, 1)
        X = np.random.randn(50, n_input)
        y = np.random.randn(50, n_output)
        st.markdown("### Random Dataset Preview")
        st.dataframe(pd.DataFrame(np.hstack([X, y]), columns=[f"X{i+1}" for i in range(X.shape[1])] + [f"Y{i+1}" for i in range(y.shape[1])]))

    # ================== Model Configuration ==================
    st.markdown("### ⚙️ Model Configuration")
    n_hidden_layers = st.slider("Hidden Layers", 0, 5, 1)
    hidden_neurons = []
    for i in range(n_hidden_layers):
        hidden_neurons.append(st.slider(f"Neurons in Layer {i+1}", 1, 20, 4))

    activation = st.selectbox("Activation Function", ["Sigmoid", "Tanh", "Linear"])
    lr = st.slider("Learning Rate", 0.001, 0.1, 0.01)
    epochs = st.slider("Epochs", 10, 500, 100, step=10)

    layers = [X.shape[1]] + hidden_neurons + [y.shape[1]]

    # ================== Train Network ==================
    if st.button("🚀 Train Neural Network"):
        with st.spinner("Training the network..."):
            weights, biases, losses = train_network(X, y, layers, lr, epochs, activation)

        st.success("✅ Training Completed!")

        # ================== Plots ==================
        st.markdown("### 📉 Loss Curve")
        st.pyplot(plot_loss(losses))

        st.markdown("### 🖼 Network Architecture")
        st.pyplot(network_diagram(layers))

        # ================== Step-by-step Math ==================
        st.markdown("### 🧮 Step-by-step Math for First Sample")
        steps = mlp_math_steps(X[:1], y[:1], weights, biases, activation)
        for s in steps:
            st.text(s)

        st.markdown("---")
        st.markdown("🎯 **Now you can explore changing layers, neurons, LR, epochs, or dataset to see live changes!**")
