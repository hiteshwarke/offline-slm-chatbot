"""
Offline SLM Chatbot (app.py)
--------------------------------
A simple Streamlit app demonstrating an offline chatbot using a Small Language Model (SLM).
This file is intentionally documented with DETAILED comments so you (or a recruiter) can easily
understand each step. Keep in mind: model names used here are examples — you can replace them
with any compatible causal LM from Hugging Face.

How to run:
1. Install dependencies: pip install -r requirements.txt
2. Run: streamlit run app.py

Notes:
- This app tries to be CPU friendly but will use CUDA if available.
- If a model download is large, it may take time on first run.
"""

import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# ---------------------------
# Configuration / Defaults
# ---------------------------
# Default model to use. This is a "small language model" example. Replace if you prefer another SLM.
DEFAULT_MODEL = "microsoft/phi-2"  # example SLM — swap to a small model you know is available
# Model generation defaults
DEFAULT_MAX_NEW_TOKENS = 150
DEFAULT_TEMPERATURE = 0.7
DEFAULT_TOP_P = 0.9
DEFAULT_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ---------------------------
# Helper: Load model & tokenizer
# ---------------------------
# Use Streamlit cache_resource so model/tokenizer are loaded once per session and not reloaded on every interaction.
@st.cache_resource(show_spinner=False)
def load_model_and_tokenizer(model_name: str, device: str = DEFAULT_DEVICE):
    """
    Load tokenizer and model from Hugging Face.
    - model_name: HF model identifier (e.g., "microsoft/phi-2")
    - device: "cpu" or "cuda"
    Returns tuple (tokenizer, model, device)
    """
    # Informative logging for the user
    st.info(f"Loading model `{model_name}` on device `{device}`. This may take a while on first run.")
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # Load model
    model = AutoModelForCausalLM.from_pretrained(model_name)
    # Move model to device (CPU or CUDA)
    model.to(device)
    # Put model in evaluation mode (no dropout, faster inference)
    model.eval()
    return tokenizer, model, device

# ---------------------------
# Streamlit UI - Page layout
# ---------------------------
st.set_page_config(page_title="Offline SLM Chatbot", layout="wide")
st.title("🤖 Offline SLM Chatbot (Small Language Model)")

# Sidebar: settings that user can tweak
with st.sidebar:
    st.header("⚙️ Settings")
    model_name = st.text_input("Model name (Hugging Face)", value=DEFAULT_MODEL)
    device_choice = st.selectbox("Device", options=["cuda", "cpu"], index=0 if DEFAULT_DEVICE=="cuda" else 1)
    max_new_tokens = st.number_input("Max new tokens", min_value=16, max_value=1024, value=DEFAULT_MAX_NEW_TOKENS, step=16)
    temperature = st.slider("Temperature (creativity)", min_value=0.0, max_value=1.5, value=float(DEFAULT_TEMPERATURE))
    top_p = st.slider("Top-p (nucleus sampling)", min_value=0.1, max_value=1.0, value=float(DEFAULT_TOP_P))
    clear_history = st.button("Clear chat history")

# Load model (cached). If loading fails, show error.
try:
    tokenizer, model, model_device = load_model_and_tokenizer(model_name, device_choice)
except Exception as e:
    st.error(f"Failed to load model `{model_name}`. Error: {e}")
    st.stop()

# ---------------------------
# Chat history state
# ---------------------------
# We keep a simple chat history in Streamlit session_state so that conversation persists between reruns.
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []  # list of tuples (role, text)

if clear_history:
    st.session_state.chat_history = []
    st.success("Chat history cleared.")

# ---------------------------
# Chat input area
# ---------------------------
# Provide a friendly example prompt to help first-time users.
st.markdown("Write a message below and press Enter. Try: *'Explain reinforcement learning in simple terms.'*")

user_input = st.text_input("You:", key="user_input_text")

# When the user enters text, generate a response using the model
if user_input:
    # Append user message to history
    st.session_state.chat_history.append(("User", user_input))

    # Prepare the prompt for the causal LM.
    # For simple chat, we can pass the latest user message as the prompt.
    # For more sophisticated chat, you might build a dialogue context with previous messages.
    prompt = user_input

    # Tokenize input prompt (return tensor)
    inputs = tokenizer(prompt, return_tensors="pt").to(model_device)

    # Generate with the model using the configured generation params
    with st.spinner("Generating response..."):
        # We use no_grad context to save memory
        with torch.no_grad():
            outputs = model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs.get("attention_mask", None),
                max_new_tokens=int(max_new_tokens),
                temperature=float(temperature),
                top_p=float(top_p),
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id if tokenizer.eos_token_id is not None else tokenizer.pad_token_id
            )
    # Decode the generated tokens to text
    response_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # If the model echoes the prompt, attempt to trim the prompt from the response for clarity
    if response_text.startswith(prompt):
        response_text = response_text[len(prompt):].strip()

    # Append bot response to history
    st.session_state.chat_history.append(("Bot", response_text))

# ---------------------------
# Display chat history
# ---------------------------
# Show messages in order
for role, message in st.session_state.chat_history:
    if role == "User":
        st.markdown(f"**You:** {message}")
    else:
        # Bot message
        st.markdown(f"**Bot:** {message}")

# ---------------------------
# Helpful notes for the user (explainability)
# ---------------------------
st.markdown("---")
st.subheader("Notes & Tips")
st.markdown("""
- **Offline**: This app runs **locally** and does not send your data to external APIs.
- **Replace model**: If `microsoft/phi-2` is not available or too large, change the model name in the sidebar to a smaller model such as `distilgpt2` or `EleutherAI/gpt-neo-125M`.
- **Performance**: Small models run fast on CPU; larger models may require more RAM or GPU.
- **Improvements**: For a better chat experience, you can add context-window management (prepend recent messages to the prompt) or fine-tune a small model on conversational data.
""")