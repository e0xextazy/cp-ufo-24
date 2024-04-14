import streamlit as st
import pandas as pd

st.set_page_config(
    page_title="Home Page",
    page_icon="🏠",
)

if "data" not in st.session_state:
    st.session_state["data"] = pd.DataFrame(
        columns=["Class", "Summary", "filename"])

st.write("# Welcome to 5️⃣Random's Demo")

st.sidebar.success("Select a mode above.")

st.markdown(
    """
    **Мы реализовали следующий функционал:**
    - Загрузка документов
      - .rtf
      - .doc
      - .pdf
      - .txt
    - Классификация документов на 11 типов
      - Доверенность
      - Договор
      - Акт
      - Заявление
      - Приказ
      - Счет
      - Приложение
      - Соглашение
      - Договор оферты
      - Устав
      - Решение
    - Предсказание заголовка по содержанию файла
    - Интерактивная таблица с умной фильтрацией
      - Расширение файла
      - Размер файла
      - Тип документа
      - Дата добавления
      - Поиск по вхождению слова
    """
)
