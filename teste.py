import streamlit as st
import requests

st.image("img-grupo.jpeg", width=800)  # URL de exemplo

st.title("Gerenciador de Grupo de WhatsApp")

if "upload_done" not in st.session_state:
    st.session_state.upload_done = False

# Criar containers para controle dinâmico
upload_container = st.empty()
ia_container = st.empty()

# Renderizar upload se ainda não foi feito
if not st.session_state.upload_done:
    with upload_container.form(key="upload_form"):
        st.subheader("Faça Upload da sua conversa do WhatsApp para o banco")
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
                        try:
                            result = response.json()
                            st.success(f"Arquivo enviado com sucesso!")
                        except Exception:
                            st.success(f"Arquivo enviado com sucesso! Resposta da API: {response.text}")

                        # Marcar upload como feito e atualizar interface
                        st.session_state.upload_done = True
                        upload_container.empty()  # Remove o formulário de upload
                    else:
                        st.error(f"Erro ao enviar arquivo: {response.status_code} - {response.text}")
                except Exception as e:
                    st.error(f"Erro de conexão com API: {e}")
            else:
                st.warning("Selecione um arquivo antes de enviar.")

# Mostrar formulário da IA se upload foi feito
if st.session_state.upload_done:
    with ia_container.form(key="ia_form"):
        st.subheader("Faça sua pergunta:")
        user_input = st.text_area("Digite algo:", "")
        submit_ia = st.form_submit_button("Enviar para IA")

        if submit_ia:
            if user_input.strip() != "":
                try:
                    response = requests.post(
                        "http://localhost:3000/ia-prompt",
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
