# visuals/plots.py
import matplotlib.pyplot as plt
import streamlit as st

def plot_loss_curve(losses):
    fig, ax = plt.subplots()
    ax.plot(losses, linewidth=2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE Loss")
    ax.set_title("Loss Curve")
    st.pyplot(fig)

def plot_actual_vs_pred(y_true, y_pred):
    fig, ax = plt.subplots()
    ax.scatter(y_true, y_pred, s=40, color='#2563eb')
    mn, mx = min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())
    ax.plot([mn, mx], [mn, mx], "r--", linewidth=2)
    ax.set_xlabel("Actual")
    ax.set_ylabel("Predicted")
    ax.set_title("Actual vs Predicted")
    st.pyplot(fig)
