# visuals/diagrams.py
import networkx as nx
import matplotlib.pyplot as plt
import streamlit as st

def draw_network(layers):
    """
    Draws a simple feedforward network diagram.
    layers: list of layer sizes, e.g. [3,4,2]
    """
    G = nx.DiGraph()
    pos = {}
    x = 0
    for l, size in enumerate(layers):
        ys = list(range(-size, size*2, 2))
        for i in range(size):
            node = f"L{l}N{i}"
            G.add_node(node)
            pos[node] = (x, ys[i])
        x += 2.5
    for l in range(len(layers)-1):
        for i in range(layers[l]):
            for j in range(layers[l+1]):
                G.add_edge(f"L{l}N{i}", f"L{l+1}N{j}")

    fig, ax = plt.subplots(figsize=(8,6))
    nx.draw(G, pos, node_color="#93c5fd", node_size=700, with_labels=True, ax=ax)
    st.pyplot(fig)
