import streamlit as st
from langchain_community.llms import Ollama

# --- Page Configuration ---
st.set_page_config(
    page_title="Local LLM Chatbot",
    page_icon="🤖",
    layout="wide"
)

# --- Sidebar Configuration ---
st.sidebar.title("🤖 Local LLM Chatbot")
st.sidebar.markdown("Choose your model and settings.")

# Model Selection
model_name = st.sidebar.selectbox(
    "Choose a model",
    ("llama3:8b", "gemma:7b")
)

# Advanced Settings
with st.sidebar.expander("⚙️ Advanced Settings"):
    temperature = st.slider("Temperature", min_value=0.0, max_value=2.0, value=0.7, step=0.1,
                            help="Controls randomness. Lower is more deterministic.")
    top_p = st.slider("Top P", min_value=0.1, max_value=1.0, value=0.9, step=0.1,
                      help="Filters vocabulary to the most probable tokens.")
    top_k = st.number_input("Top K", min_value=1, max_value=100, value=40,
                            help="Filters vocabulary to the top K most probable tokens.")
    max_tokens = st.number_input("Max Tokens", min_value=64, max_value=4096, value=512,
                                 help="Maximum number of tokens to generate.")

# --- Summarization Feature ---
if st.sidebar.button("Summarize Chat"):
    if len(st.session_state.messages) > 1:
        with st.spinner("Summarizing..."):
            conversation = "\n".join(
                [f"{msg['role']}: {msg['content']}" for msg in st.session_state.messages]
            )
            summarization_prompt = f"Please provide a concise summary of the following conversation:\n\n{conversation}"
            summary = Ollama(model=model_name).invoke(summarization_prompt) # Use a separate instance for summary
            st.sidebar.subheader("Chat Summary")
            st.sidebar.info(summary)
    else:
        st.sidebar.warning("Not enough conversation to summarize.")
        
# A button to clear the chat history
if st.sidebar.button("Clear Chat History"):
    st.session_state.messages = []
    st.rerun()


# --- Model Initialization ---
@st.cache_resource
def get_llm(model, temp, p, k, max_tok):
    return Ollama(
        model=model,
        temperature=temp,
        top_p=p,
        top_k=k,
        num_predict=max_tok,
    )

llm = get_llm(model_name, temperature, top_p, top_k, max_tokens)

# --- Main Chat Interface ---
st.title(f"Chat with {model_name}")
st.markdown("This is a local chatbot powered by Ollama. Start chatting below!")

# Chat History Management
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat Input and Response Generation
if prompt := st.chat_input("What would you like to ask?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        response_stream = llm.stream(prompt)
        full_response = st.write_stream(response_stream)
    
    st.session_state.messages.append({"role": "assistant", "content": full_response})