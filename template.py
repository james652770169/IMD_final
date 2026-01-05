import streamlit as st

st.set_page_config(page_title="國立虎尾科技大學機械設計工程系", layout="wide")



st.title("114(上)『智慧機械設計』課程期末報告")

st.title("RO機器人與AI整合之研究")

st.header("指導老師：周榮源")
st.header("班級：四設計一甲")
st.header("組別：第六組")
st.header("組員：11473103/林敬傑")

st.write("歡迎來到RO機器人與AI整合之研究報告。請從左側選單選擇要查看的實驗項目。")

# 不需要在這裡定義實驗項目列表，因為Streamlit會自動生成側邊欄

# 顯示一張圖片(image)
st.image("files/NFU.png")
# Caption
st.caption("""(這是國立虎尾科技大學之校門!)記得在影片或是照片後加上圖例說明!""")



