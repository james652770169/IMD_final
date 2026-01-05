import streamlit as st
st.set_page_config(page_title="國立虎尾科技大學機械設計工程系", layout="wide")


st.title("第五章 結論與心得")

st.markdown('### 步驟1：安裝streamlit套件')
code = """
            pip install streamlit
            """
st.code(code, language="python")

st.markdown('### 步驟2：啟動streamlit，開啟網頁')
code = """
            STREAMLIT run template.py
            """
st.code(code, language="python")

st.image("Picture/第4章01.png")

st.markdown('### 步驟3：開啟"[streamlit](https://cheat-sheet.streamlit.app/)"官網的程式庫，從中可獲的各種程式以供網頁書寫') 

