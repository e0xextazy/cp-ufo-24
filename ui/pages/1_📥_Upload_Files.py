import streamlit as st
from striprtf.striprtf import rtf_to_text
from io import StringIO
from datetime import date

from models import get_predicts

st.set_page_config(page_title="Upload files", page_icon="📥")


def get_text(uploaded_file):
    if uploaded_file.type == "application/rtf":
        text = StringIO(uploaded_file.getvalue().decode("utf-8"))
        text = text.read()
        text = rtf_to_text(text)
    elif uploaded_file.type == "application/doc":
        pass
    elif uploaded_file.type == "application/pdf":
        pass
    elif uploaded_file.type == "application/txt":
        pass
    else:
        raise TypeError("Invalid type")

    return text


st.markdown("# Upload Files")
st.sidebar.header("Upload Files")

progress_bar = st.sidebar.progress(0)
status_text = st.sidebar.empty()
data = []

with st.form("my-form", clear_on_submit=True):
    uploaded_files = st.file_uploader(
        "Choose a files", accept_multiple_files=True, type=["rtf", "doc", "pdf", "txt"])
    num_files = len(uploaded_files)
    submitted = st.form_submit_button("Upload")
    if submitted:
        write = st.form_submit_button("Write to DataBase")
        if write:
            for i in data:
                st.session_state["data"].loc[len(
                    st.session_state["data"].index)] = data[i]

if uploaded_files is not None:
    for i, uploaded_file in enumerate(uploaded_files):
        status_text.text(f"{i + 1}/{num_files}")
        progress_bar.progress((i + 1) / num_files)

        text = get_text(uploaded_file)
        bytes_data = uploaded_file.getvalue()

        class_pred, summary_pred = get_predicts(text)
        file_size = len(bytes_data)
        current_date = date.today()
        ext = uploaded_file.type.split("/")[-1]
        filename = uploaded_file.name
        data.append([class_pred, summary_pred, filename])

        st.markdown(f'''
                    - **{filename}**
                      - **Тип документа:** {class_pred};
                      - **Заголовок:** {summary_pred}.
                    ''')
        progress_bar.empty()
