import streamlit as st
st.set_page_config(page_title="國立虎尾科技大學機械設計工程系", layout="wide")

st.title("第三章 YOLO影像辨識")

st.markdown(' 3.1：確認要辨認的物體，本題目依題目設定目標物體為杏鮑菇、番茄、蘋果')
st.markdown(' 3.2：三個物體各尋找樣本照片')
st.markdown(' 3.3：步驟3：標註圖片，可使用線上標註網站"[roboflow](https://app.roboflow.com/test-6swly)"')
st.image("Picture/第3章01.png")
st.image("Picture/第3章02.png")
st.image("Picture/第3章03.png")
st.markdown(' 3.4：撰寫一個由yolo11n訓練的模型，訓練的目標為辨識杏鮑菇、番茄與蘋果的')