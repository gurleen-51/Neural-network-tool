import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sns

import networkx as nx
import matplotlib.pyplot as plt
import time
from core.network import train_network
from visuals.plots import plot_loss_curve, plot_actual_vs_pred
from visuals.diagrams import draw_network
from math_utils.perceptron_math import perceptron_math_steps
from math_utils.mlp_math import mlp_math_steps


# ================= CSS =================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Lobster&display=swap');

.neural-header {
    font-family: 'Lobster', cursive;
    font-size: 70px;
    text-align: center;
    background: linear-gradient(90deg, #800000, #a52a2a, #b22222); /* maroon shades */
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
</style>

<div class="neural-header">Neural Network</div>
""", unsafe_allow_html=True)




def draw_network_diagram(layers, activations=None, weights=None):
    """
    Draws the neural network diagram with activations and weights.
    Returns the matplotlib figure object for Streamlit.
    """
    G = nx.DiGraph()
    pos = {}
    x_spacing = 3
    node_labels = {}
    edge_labels = {}

    # Add Nodes
    for l, size in enumerate(layers):
        y_positions = np.linspace(-size+1, size-1, size)
        for i in range(size):
            node_name = f"L{l}N{i}"
            G.add_node(node_name)
            pos[node_name] = (l*x_spacing, y_positions[i])
            if activations and l < len(activations):
                node_labels[node_name] = f"{activations[l][0,i]:.2f}"
            else:
                node_labels[node_name] = node_name

    # Add Edges
    if weights:
        for l in range(len(layers)-1):
            for i in range(layers[l]):
                for j in range(layers[l+1]):
                    G.add_edge(f"L{l}N{i}", f"L{l+1}N{j}")
                    edge_labels[(f"L{l}N{i}", f"L{l+1}N{j}")] = f"{weights[l][i,j]:.2f}"
    else:
        for l in range(len(layers)-1):
            for i in range(layers[l]):
                for j in range(layers[l+1]):
                    G.add_edge(f"L{l}N{i}", f"L{l+1}N{j}")

    # Draw network on a new figure
    fig, ax = plt.subplots(figsize=(8,6))
    nx.draw(G, pos, node_size=1000, node_color="#93c5fd", with_labels=False, ax=ax)
    nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=10, font_color='black', ax=ax)
    if weights:
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='red', font_size=8, ax=ax)
    ax.axis('off')
    plt.tight_layout()
    
    return fig  # Return the figure object


# ================= PAGE CONFIG =================
st.set_page_config(page_title="Neural Network Visual Lab", layout="wide")

# ================= SESSION STATE =================
st.sidebar.title("⚙️ Navigation")
if st.sidebar.button("🏠 Home"):
    st.session_state.page = "home"
if st.sidebar.button("🧠 Playground"):
    st.session_state.page = "playground"
if st.sidebar.button("📚 Knowledge Base"):
    st.session_state.page = "knowledge"
if st.sidebar.button("📐 Learn the Math"):
    st.session_state.page = "math"
if st.sidebar.button("📊 Visual Learning"):
    st.session_state.page = "visual"





# ================= LANDING PAGE =================
def landing_page():
   
    st.markdown("<p style='text-align:center;color:#6b7280;'>Learn Neural Networks visually, mathematically, and intuitively</p>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("📐 Learn the Math"): st.session_state.page="math"
        st.markdown('<div class="card"><div class="card-title">📐 Learn the Math</div><div class="card-desc">Step-by-step math for Perceptron & MLP</div></div>', unsafe_allow_html=True)
    with col2:
        if st.button("🧠 Playground"): st.session_state.page="playground"
        st.markdown('<div class="card"><div class="card-title">🧠 Build Networks</div><div class="card-desc">Upload dataset, train & experiment</div></div>', unsafe_allow_html=True)
    with col3:
        if st.button("📊 Visualize Learning"): st.session_state.page="visual"
        st.markdown('<div class="card"><div class="card-title">📊 Visualize Learning</div><div class="card-desc">View network, loss & predictions</div></div>', unsafe_allow_html=True)

# ================= KNOWLEDGE BASE =================
def knowledge_base_section():
    st.markdown("<h2>📚 Knowledge Base: Deep Learning & ANN</h2>", unsafe_allow_html=True)
    st.markdown("""
### 1️⃣ Introduction
Artificial Neural Networks (ANNs) are computational models inspired by the human brain. They can learn patterns from data and make predictions.

### 2️⃣ History & Literature
- Introduced in 1958 (Rosenblatt Perceptron)  
- MLP and Backpropagation (Rumelhart, 1986)  
- Modern deep learning: Goodfellow et al., 2016  

### 3️⃣ Types of Neural Networks
- Perceptron (single layer)  
- Multi-Layer Perceptron (MLP)  
- Convolutional Neural Networks (CNN)  
- Recurrent Neural Networks (RNN)

### 4️⃣ Activation Functions
- Sigmoid: 0→1, smooth gradient  
- Tanh: -1→1, zero-centered  
- Linear: identity function

### 5️⃣ Loss & Optimization
- MSE, Cross-Entropy  
- Gradient Descent, SGD, Adam

### 6️⃣ Backpropagation
- Compute derivative of loss w.r.t weights  
- Update weights layer by layer  

### 7️⃣ Real-World Applications
- Image classification, speech recognition  
- NLP, autonomous driving, finance predictions

### 8️⃣ References
- Goodfellow, I., Bengio, Y., & Courville, A. (2016). Deep Learning. MIT Press
- Rumelhart, Hinton & Williams (1986). Learning representations by backpropagation
""")

# ================= LEARN MATH =================
def learn_math_page():
    st.markdown("<h2>📐 Learn the Math</h2>", unsafe_allow_html=True)
    st.markdown("Interactive step-by-step calculations with visualizations.")

    # Number of inputs and neurons
    n_features = st.number_input("Number of input features", 1, 5, 2)
    X_vals = st.text_input(f"Enter {n_features} input values separated by comma", "0.5, 1.0")
    X = np.array([list(map(float, X_vals.split(",")))]).reshape(1, n_features)

    n_neurons = st.number_input("Number of neurons in output layer", 1, 3, 1)
    W_vals = st.text_input(f"Enter {n_features*n_neurons} weight values row-wise separated by comma",
                           ",".join(["0.4"]* (n_features*n_neurons)))
    W = np.array(list(map(float, W_vals.split(",")))).reshape(n_features, n_neurons)

    b_vals = st.text_input(f"Enter {n_neurons} bias values separated by comma", ",".join(["0.1"]*n_neurons))
    b = np.array([list(map(float, b_vals.split(",")))])

    activation = st.selectbox("Activation Function", ["Sigmoid", "Tanh", "Linear"])

    # ---------------- Compute ----------------
    def activate(z):
        if activation=="Sigmoid": return 1/(1+np.exp(-z))
        elif activation=="Tanh": return np.tanh(z)
        else: return z
    def activate_derivative(z):
        if activation=="Sigmoid":
            a = 1/(1+np.exp(-z)); return a*(1-a)
        elif activation=="Tanh": return 1-np.tanh(z)**2
        else: return np.ones_like(z)

    z = X @ W + b
    a = activate(z)
    error = 0.5*(a)**2
    dz = activate_derivative(z)

    st.markdown("### Step 1: Linear Combination")
    st.latex(r"z = X \cdot W + b")
    st.write(f"Substitute: z = {z}")

    st.markdown("### Step 2: Activation")
    st.latex(r"a = activation(z)")
    st.write(f"a = {a}")

    st.markdown("### Step 3: Error")
    st.latex(r"Error = 0.5*(a)^2")
    st.write(f"Error = {error}")

    st.markdown("### Step 4: Derivative w.r.t z")
    st.write(f"Derivative = {dz}")

    # ---------------- Network Visualization ----------------
    st.markdown("### 🖼 Network Diagram")
    draw_network([n_features, n_neurons])

def playground_section():
    st.markdown("<h2>🧠 Playground: Train & Explore</h2>", unsafe_allow_html=True)
    st.markdown("Upload dataset or use random, configure your network, and train interactively.")

    # ---------------- Dataset ----------------
    use_csv = st.checkbox("Use CSV Dataset")
    if use_csv:
        uploaded_file = st.file_uploader("Upload CSV (numeric columns only)", type=["csv"])
        if uploaded_file:
            df = pd.read_csv(uploaded_file)
            st.write("Preview of dataset:")
            st.dataframe(df.head())
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            input_cols = st.multiselect("Select Input Features", numeric_cols, default=numeric_cols[:-1])
            target_col = st.selectbox("Select Target Column", numeric_cols, index=len(numeric_cols)-1)
            if input_cols and target_col:
                X = df[input_cols].values
                y = df[[target_col]].values
        else:
            st.info("No file uploaded, using random dataset")
            X = np.random.randn(50,2)
            y = np.random.randn(50,1)
    else:
        X = np.random.randn(50,2)
        y = np.random.randn(50,1)

    st.write(f"Dataset shape X:{X.shape}, y:{y.shape}")

    # ---------------- Network Configuration ----------------
    st.markdown("### ⚙️ Network Configuration")
    model_type = st.selectbox("Model Type", ["Perceptron", "Multi-Layer Perceptron"])
    activation = st.selectbox("Activation Function", ["Sigmoid","Tanh","Linear"])
    lr = st.number_input("Learning Rate", 0.001, 0.05, 0.01)
    epochs = st.number_input("Epochs", 10, 1000, 100, step=10)

    hidden_neurons=[]
    if model_type=="Multi-Layer Perceptron":
        n_hidden_layers = st.number_input("Number of Hidden Layers", 0, 5, 1)
        for i in range(n_hidden_layers):
            hidden_neurons.append(st.number_input(f"Neurons in Hidden Layer {i+1}", 1, 30, 4))

    # ---------------- TRAIN BUTTON ----------------
    

        # ---------------- Initialize ----------------
        layers = [X.shape[1]] + hidden_neurons + [y.shape[1]] if model_type=="Multi-Layer Perceptron" else [X.shape[1], y.shape[1]]
        weights = [np.random.randn(layers[i], layers[i+1])*0.1 for i in range(len(layers)-1)]
        biases = [np.zeros((1,layers[i+1])) for i in range(len(layers)-1)]
        losses = []

        def act_fn(z):
            if activation=="Sigmoid": return 1/(1+np.exp(-z))
            elif activation=="Tanh": return np.tanh(z)
            else: return z

        def act_deriv(z):
            if activation=="Sigmoid":
                a = 1/(1+np.exp(-z))
                return a*(1-a)
            elif activation=="Tanh": return 1-np.tanh(z)**2
            else: return np.ones_like(z)

        # Normalize
        X_norm = (X - X.mean(axis=0)) / (X.std(axis=0)+1e-8)

        # ---------------- Training Loop ----------------
        for epoch in range(epochs):
            a_list = [X_norm]
            z_list = []

            # Forward
            for l in range(len(weights)):
                z = a_list[-1] @ weights[l] + biases[l]
                z_list.append(z)
                a_list.append(act_fn(z))

            # Loss
            mse = np.mean((y - a_list[-1])**2)
            losses.append(mse)

            # Backprop
            delta = (a_list[-1]-y)*act_deriv(z_list[-1])
            deltas=[delta]
            for l in range(len(weights)-2,-1,-1):
                delta = deltas[0] @ weights[l+1].T * act_deriv(z_list[l])
                deltas.insert(0, delta)
            for l in range(len(weights)):
                weights[l] -= lr*(a_list[l].T @ deltas[l])/X.shape[0]
                biases[l] -= lr*np.mean(deltas[l],axis=0,keepdims=True)

            progress.progress((epoch+1)/epochs)
            time.sleep(0.005)

        st.success("✅ Training Completed!")
        st.write(f"Final MSE: {losses[-1]:.4f}")
        
        # ---------------------- Playground Section ----------------------
def playground_section():
    st.markdown("<h2>🧠 Playground: Train & Explore</h2>", unsafe_allow_html=True)
    st.markdown("Upload dataset or use random data, configure your network, and explore the training visually.")

    # ---------------- Dataset ----------------
    use_csv = st.checkbox("Use CSV Dataset")
    if use_csv:
        uploaded_file = st.file_uploader("Upload CSV (numeric columns only)", type=["csv"])
        if uploaded_file:
            df = pd.read_csv(uploaded_file)
            st.write("Preview of dataset:")
            st.dataframe(df.head())
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            input_cols = st.multiselect("Select Input Features", numeric_cols, default=numeric_cols[:-1])
            target_col = st.selectbox("Select Target Column", numeric_cols, index=len(numeric_cols)-1)
            if input_cols and target_col:
                X = df[input_cols].values
                y = df[[target_col]].values
                st.markdown(f"**Dataset Info:** {X.shape[0]} samples, {X.shape[1]} features, 1 target")
                st.write(f"Input mean/std: {X.mean(axis=0)}, {X.std(axis=0)}")
        else:
            st.info("No CSV uploaded, using random dataset")
            X = np.random.randn(50,2)
            y = np.random.randn(50,1)
    else:
        X = np.random.randn(50,2)
        y = np.random.randn(50,1)
        st.markdown(f"**Dataset Info:** {X.shape[0]} samples, {X.shape[1]} features, 1 target")
        st.write(f"Input mean/std: {X.mean(axis=0)}, {X.std(axis=0)}")

    # ---------------- Network Config ----------------
    st.markdown("### ⚙️ Network Configuration")
    model_type = st.selectbox("Model Type", ["Perceptron", "Multi-Layer Perceptron"])
    activation = st.selectbox("Activation Function", ["Sigmoid","Tanh","Linear"])
    lr = st.number_input("Learning Rate", 0.001, 0.05, 0.01)
    epochs = st.number_input("Epochs", 10, 500, 100, step=10)

    hidden_neurons=[]
    if model_type=="Multi-Layer Perceptron":
        n_hidden_layers = st.number_input("Number of Hidden Layers", 0, 5, 1)
        for i in range(n_hidden_layers):
            hidden_neurons.append(st.number_input(f"Neurons in Hidden Layer {i+1}", 1, 30, 4))

    # ---------------- Train Button ----------------
    if st.button("🚀 Train Network"):
        st.info("Training started...")
        progress = st.progress(0)

        # Layers
        layers = [X.shape[1]] + hidden_neurons + [y.shape[1]] if model_type=="Multi-Layer Perceptron" else [X.shape[1], y.shape[1]]
        weights = [np.random.randn(layers[i], layers[i+1])*0.1 for i in range(len(layers)-1)]
        biases = [np.zeros((1,layers[i+1])) for i in range(len(layers)-1)]
        losses = []

        def act_fn(z):
            if activation=="Sigmoid": return 1/(1+np.exp(-z))
            elif activation=="Tanh": return np.tanh(z)
            else: return z

        def act_deriv(z):
            if activation=="Sigmoid":
                a = 1/(1+np.exp(-z))
                return a*(1-a)
            elif activation=="Tanh": return 1-np.tanh(z)**2
            else: return np.ones_like(z)

        # Normalize X
        X_norm = (X - X.mean(axis=0)) / (X.std(axis=0)+1e-8)

        # ---------------- Training Loop ----------------
        for epoch in range(epochs):
            a_list = [X_norm]
            z_list = []

            # Forward
            for l in range(len(weights)):
                z = a_list[-1] @ weights[l] + biases[l]
                z_list.append(z)
                a_list.append(act_fn(z))

            # Loss
            mse = np.mean((y - a_list[-1])**2)
            losses.append(mse)

            # Backprop
            delta = (a_list[-1]-y)*act_deriv(z_list[-1])
            deltas=[delta]
            for l in range(len(weights)-2,-1,-1):
                delta = deltas[0] @ weights[l+1].T * act_deriv(z_list[l])
                deltas.insert(0, delta)
            for l in range(len(weights)):
                weights[l] -= lr*(a_list[l].T @ deltas[l])/X.shape[0]
                biases[l] -= lr*np.mean(deltas[l],axis=0,keepdims=True)

            progress.progress((epoch+1)/epochs)
        
        st.success("✅ Training Completed!")
        st.write(f"Final MSE: {losses[-1]:.4f}")
        st.markdown("**Network learned to minimize error using backpropagation and gradient descent.**")

        # ---------------- Visualization (Compact) ----------------
        fig, ax = plt.subplots(figsize=(6,4))
        ax.plot(losses, linewidth=2, color="#2563eb")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("MSE")
        ax.set_title("📉 Loss Curve")
        st.pyplot(fig)

        fig, ax = plt.subplots(figsize=(6,4))
        ax.scatter(y, a_list[-1], color="#f43f5e", s=40)
        mn, mx = min(y.min(), a_list[-1].min()), max(y.max(), a_list[-1].max())
        ax.plot([mn,mx],[mn,mx],'r--',linewidth=1.5)
        ax.set_xlabel("Actual")
        ax.set_ylabel("Predicted")
        ax.set_title("📊 Actual vs Predicted")
        st.pyplot(fig)

        st.subheader("🖼 Network Diagram")
        fig = draw_network_diagram(layers, activations=a_list, weights=weights)
        st.pyplot(fig)

        # Weights & Biases table
        st.subheader("📝 Final Weights & Biases")
        for idx,(w,bias) in enumerate(zip(weights,biases)):
            st.markdown(f"**Layer {idx+1} weights:**")
            st.write(w)
            st.markdown(f"**Layer {idx+1} biases:**")
            st.write(bias)



# ================= VISUAL LEARNING =================
def visual_learning_section():
    st.markdown("<h2>📊 Visual Learning</h2>", unsafe_allow_html=True)
    st.markdown("Upload a dataset or use random data to explore your neural network visually and understand learning.")

    # ---------------- Dataset ----------------
    use_csv = st.checkbox("Use CSV Dataset")
    if use_csv:
        uploaded_file = st.file_uploader("Upload CSV (numeric columns only)", type=["csv"])
        if uploaded_file:
            df = pd.read_csv(uploaded_file)
            st.write("Preview of dataset:")
            st.dataframe(df.head())

            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            input_cols = st.multiselect("Select Input Features", numeric_cols, default=numeric_cols[:-1])
            target_col = st.selectbox("Select Target Column", numeric_cols, index=len(numeric_cols)-1)

            if input_cols and target_col:
                X = df[input_cols].values
                y = df[[target_col]].values
        else:
            st.info("No CSV uploaded, using random dataset")
            X = np.random.randn(50,2)
            y = np.random.randn(50,1)
    else:
        X = np.random.randn(50,2)
        y = np.random.randn(50,1)

    st.write(f"Dataset shape X:{X.shape}, y:{y.shape}")

    # ---------------- Feature Analysis ----------------
    st.subheader("🔹 Feature Correlation Heatmap")
    df_vis = pd.DataFrame(np.hstack([X,y]), columns=[f"X{i}" for i in range(X.shape[1])] + ["Y"])
    fig, ax = plt.subplots(figsize=(5,4))
    sns.heatmap(df_vis.corr(), annot=True, cmap="coolwarm", ax=ax)
    st.pyplot(fig)

    st.subheader("🔹 Pairplot of Features")
    fig2 = sns.pairplot(df_vis)
    st.pyplot(fig2)

    # ---------------- Simple Network Simulation ----------------
    st.subheader("⚙️ Simple Neural Network Example")
    n_input = X.shape[1]
    n_output = y.shape[1]

    W = np.random.rand(n_input, n_output)
    b = np.random.rand(1, n_output)

    st.markdown("Randomly initialized weights and biases for demonstration:")
    st.write("Weights:\n", W)
    st.write("Biases:\n", b)

    # Forward pass
    a = 1 / (1 + np.exp(-(X @ W + b)))
    st.markdown("Forward Pass Activations (using Sigmoid):")
    st.write(a[:5,:])  # show first 5 rows

    # ---------------- Network Diagram ----------------
    st.subheader("🖼 Network Diagram")
    G = nx.DiGraph()
    pos = {}
    x_spacing = 3
    for i in range(n_input):
        G.add_node(f"I{i}", layer="input")
        pos[f"I{i}"] = (0, i)
    for j in range(n_output):
        G.add_node(f"O{j}", layer="output")
        pos[f"O{j}"] = (x_spacing, j)
    for i in range(n_input):
        for j in range(n_output):
            G.add_edge(f"I{i}", f"O{j}", weight=W[i,j])

    fig3, ax = plt.subplots(figsize=(6,4))
    nx.draw(G, pos, with_labels=True, node_size=1000, node_color="#93c5fd", ax=ax)
    edge_labels = {(f"I{i}", f"O{j}"): round(W[i,j],2) for i in range(n_input) for j in range(n_output)}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color="red", ax=ax)
    st.pyplot(fig3)

    # ---------------- Loss Simulation ----------------
    st.subheader("📉 Simulated Learning (MSE over Epochs)")
    epochs = 50
    losses = [np.mean((y - (1/(1+np.exp(-(X @ W + b)))))**2) * (0.99**e) for e in range(epochs)]
    fig4, ax = plt.subplots(figsize=(5,4))
    ax.plot(range(1,epochs+1), losses, marker='o', color="maroon")
    ax.set_xlabel("Epochs")
    ax.set_ylabel("MSE")
    st.pyplot(fig4)

    st.markdown("✅ Visual Learning Complete. You can now interactively explore weights, activations, and simulated loss trends.")



# ================= MAIN =================
def main():
    page = st.session_state.get("page","home")
    if page=="home": landing_page()
    elif page=="knowledge": knowledge_base_section()
    elif page=="math": learn_math_page()
    elif page=="playground": playground_section()
    elif page=="visual": visual_learning_section()
    else: landing_page()

if __name__=="__main__":
    main()
