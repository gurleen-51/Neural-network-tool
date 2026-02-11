import streamlit as st

def configure_sidebar():
    st.sidebar.title("⚙️ Configuration")
    st.sidebar.markdown("### Navigation")
    if st.sidebar.button("🏠 Home"): st.session_state.page="home"
    if st.sidebar.button("🧠 Playground"): st.session_state.page="playground"
    if st.sidebar.button("📖 Knowledge Base"): st.session_state.page="knowledge"
