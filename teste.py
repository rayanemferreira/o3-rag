import streamlit as st
import requests

st.title("📱 Gerenciador de Grupo de WhatsApp")

# --- Formulário 1: Chat com IA ---
with st.form(key="ia_form"):
    st.subheader("Chat com IA")
    user_input = st.text_area("Digite algo:", "")
    submit_ia = st.form_submit_button("Enviar para IA")

    if submit_ia:
        if user_input.strip() != "":
            try:
                response = requests.post(
                    "http://localhost:3000/ia",
                    json={"text": user_input}
                )
                if response.status_code == 200:
                    st.success("Resposta recebida:")
                    st.write(response.text)
                else:
                    st.error(f"Erro na API: {response.status_code}")
            except Exception as e:
                st.error(f"Não foi possível conectar à API: {e}")
        else:
            st.warning("Digite algum texto antes de enviar.")

st.markdown("---")

# --- Formulário 2: Upload de arquivo TXT ---
with st.form(key="upload_form"):
    st.subheader("Upload de arquivo TXT para o banco")
    uploaded_file = st.file_uploader("Escolha um arquivo TXT", type=["txt"])
    submit_file = st.form_submit_button("Enviar arquivo")

    if submit_file:
        if uploaded_file is not None:
            try:
                files = {
                    "file": (uploaded_file.name, uploaded_file.getvalue(), "text/plain")
                }
                response = requests.post("http://localhost:3000/upload", files=files)

                if response.status_code == 200:
                    result = response.json()
                    st.success(f"Arquivo enviado com sucesso! {result['chunks']} pedaços salvos.")
                else:
                    st.error(f"Erro ao enviar arquivo: {response.status_code}")
            except Exception as e:
                st.error(f"Erro de conexão com API: {e}")
        else:
            st.warning("Selecione um arquivo antes de enviar.")
