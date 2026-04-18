import streamlit as st
import pandas as pd
import numpy as np
import time

# Page Configuration
st.set_page_config(page_title="LLM Simulation Game", layout="wide")

st.title("🤖 The Token Architect: LLM Simulation")
st.markdown("""
Welcome, Architect! Your goal is to understand how LLMs 'think' by building sentences 
one token at a time. Complete the modules below to master the concepts.
""")

# --- MODULE 1: TOKENIZATION ---
st.header("1. The Tokenizer Lab")
with st.expander("Learn about Tokenization", expanded=True):
    st.write("""
    LLMs don't read words; they read **Tokens**. A token can be a whole word, a part of a word, 
    or even a single character. 
    """)
    user_input = st.text_input("Enter a sentence to 'Tokenize':", "Artificial Intelligence is fascinating.")
    
    # Mock Tokenizer logic
    tokens = user_input.split() 
    colors = ["#FFADAD", "#FFD6A5", "#FDFFB6", "#CAFFBF", "#9BF6FF", "#A0C4FF", "#BDB2FF"]
    
    st.write("### Your Tokens:")
    cols = st.columns(len(tokens))
    for i, token in enumerate(tokens):
        cols[i].markdown(f"**{token}**")
        cols[i].markdown(f'<div style="background-color:{colors[i % len(colors)]}; height:10px; border-radius:5px;"></div>', unsafe_allow_html=True)

---

# --- MODULE 2: NEXT TOKEN PREDICTION ---
st.header("2. The Prediction Engine")
st.write("LLMs are essentially giant 'Next-Token Predictors.' Given a prompt, they calculate the probability of what comes next.")

prompt = "The students went to the library to..."
st.info(f"**Current Prompt:** {prompt}")

# Mock data for demonstration
data = {
    "Token": ["study", "sleep", "eat", "dance", "read"],
    "Logit": [4.5, 1.2, -0.5, -2.0, 3.8]
}
df = pd.DataFrame(data)

# Math for Temperature
def softmax(logits, temperature=1.0):
    # Avoid division by zero
    temp = max(temperature, 0.01)
    e_x = np.exp((logits - np.max(logits)) / temp)
    return e_x / e_x.sum()

st.subheader("Control the 'Creativity' (Temperature)")
temp = st.slider("Temperature ($T$)", min_value=0.01, max_value=2.0, value=1.0, step=0.1)

# Calculate probabilities
df['Probability'] = softmax(df['Logit'], temp)

# Visualizing the distribution
st.bar_chart(df.set_index('Token')['Probability'])

st.write(f"At Temperature **{temp}**, the model is most likely to pick: **{df.iloc[df['Probability'].idxmax()]['Token']}**")

---

# --- MODULE 3: THE GENERATION CHALLENGE ---
st.header("3. The Generation Challenge")
st.write("Try to generate a coherent sentence. Be careful—if the temperature is too high, the model might 'hallucinate' or become nonsensical!")

if st.button("Generate Random Sequence"):
    placeholder = st.empty()
    full_text = ""
    choices = ["The", "robot", "learned", "how", "to", "cook", "spaghetti", "using", "quantum", "physics"]
    
    for word in choices:
        full_text += word + " "
        placeholder.markdown(f"**Output:** `{full_text}`")
        time.sleep(0.3)
    st.success("Generation Complete!")

# --- EDUCATIONAL SIDEBAR ---
with st.sidebar:
    st.title("Concepts Cheat Sheet")
    st.markdown("""
    ### **1. Tokens**
    The basic units of text. 1,000 tokens is roughly 750 words.
    
    ### **2. Temperature**
    A scaling factor used in the **Softmax** function:
    """)
    st.latex(r"P(x_i) = \frac{e^{z_i / T}}{\sum_j e^{z_j / T}}")
    st.markdown("""
    - **Low T (< 1):** Deterministic (focused).
    - **High T (> 1):** Random (creative/chaotic).
    
    ### **3. Hallucination**
    When the model predicts a high-probability token that is factually incorrect.
    """)
