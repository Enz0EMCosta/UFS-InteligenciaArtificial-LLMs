import streamlit as st
from google import genai
from google.genai import types

class ModelConfig:
    def __init__(self, model_name, temperature, max_tokens):
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens

def truncate_history(history, limit=6):
    while len(history) > limit:
        history.pop(0) 
    return history

def conversational_agent(user_input, history, config, client):
    history.append(types.Content(role="user", parts=[types.Part(text=user_input)]))
    history = truncate_history(history, limit=8)

    response = client.models.generate_content(
        model=config.model_name,
        contents=history,
        config=types.GenerateContentConfig(
            system_instruction="Aja como um assistente acadêmico prestativo.",
            temperature=config.temperature,
            max_output_tokens=config.max_tokens,
        ),
    )

    reply_text = response.text
    history.append(types.Content(role="model", parts=[types.Part(text=reply_text)]))
    
    return reply_text, history

st.set_page_config(page_title="IA - Chatbot Contextual")

with st.sidebar:
    st.title("Configurações")
    api_key_input = st.text_input("Gemini API Key", type="password")
    st.info("Obtenha a chave em: aistudio.google.com/app/apikey")

if api_key_input:
    client = genai.Client(api_key=api_key_input)
    cfg = ModelConfig(model_name="gemini-2.5-flash", temperature=0.7, max_tokens=2048)

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        role = "user" if msg.role == "user" else "assistant"
        with st.chat_message(role):
            st.markdown(msg.parts[0].text)

    if prompt := st.chat_input("Mensagem"):
        with st.chat_message("user"):
            st.markdown(prompt)
        
        resposta, novo_historico = conversational_agent(prompt, st.session_state.messages, cfg, client)
        
        with st.chat_message("assistant"):
            st.markdown(resposta)
        
        st.session_state.messages = novo_historico
else:
    st.warning("Insira a API Key na barra lateral.")
    st.stop()