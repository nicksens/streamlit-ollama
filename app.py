import streamlit as st
import os
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

st.set_page_config(
    page_title="Cloud Chatbot",
    layout="wide"
)

def get_huggingface_token():
    token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
    if not token:
        try:
            token = st.secrets["HUGGINGFACEHUB_API_TOKEN"]
        except KeyError:
            st.error("Hugging Face API token not found! Please set it in your Streamlit secrets.")
            st.stop()
    return token

HUGGINGFACE_TOKEN = get_huggingface_token()

st.sidebar.title("Cloud Chatbot")
st.sidebar.markdown("Choose your model and settings.")

model_id = st.sidebar.selectbox(
    "Choose a model",
    ("google/gemma-2-9b-it", "mistralai/Mistral-7B-Instruct-v0.2", "meta-llama/Meta-Llama-3-8B-Instruct")
)

with st.sidebar.expander("Advanced Settings"):
    temperature = st.slider("Temperature", min_value=0.1, max_value=1.0, value=0.7, step=0.1)
    top_p = st.slider("Top-P", min_value=0.1, max_value=1.0, value=0.95, step=0.05)
    max_tokens = st.number_input("Max Tokens", min_value=64, max_value=4096, value=512)

if st.sidebar.button("Summarize Chat"):
    if len(st.session_state.get("messages", [])) > 1:
        with st.spinner("Summarizing..."):
            conversation = "\n".join(
                [f"{msg['role']}: {msg['content']}" for msg in st.session_state.messages]
            )
            summarization_prompt = f"Please provide a concise summary of the following conversation:\n\n{conversation}"

            summarizer_endpoint = HuggingFaceEndpoint(
                repo_id=model_id,
                huggingface_api_token=HUGGINGFACE_TOKEN,
                temperature=0.5,
                max_new_tokens=150
            )
            llm_summarizer = ChatHuggingFace(llm=summarizer_endpoint)
            summary_response = llm_summarizer.invoke(summarization_prompt)
            
            st.sidebar.subheader("Chat Summary")
            st.sidebar.info(summary_response.content)
    else:
        st.sidebar.warning("Not enough conversation to summarize.")

st.title(f"Chat with {model_id.split('/')[-1]}")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask me anything!"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            llm_endpoint = HuggingFaceEndpoint(
                repo_id=model_id,
                huggingface_api_token=HUGGINGFACE_TOKEN,
                temperature=temperature,
                top_p=top_p,
                max_new_tokens=max_tokens
            )
            llm = ChatHuggingFace(llm=llm_endpoint)
            
            response = llm.invoke(prompt)
            response_content = response.content
            
            st.markdown(response_content)
    
    st.session_state.messages.append({"role": "assistant", "content": response_content})
