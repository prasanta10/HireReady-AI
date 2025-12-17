import streamlit as st

home_page = st.Page("home.py", title="Home", url_path="home")
tailor_page = st.Page("tailor.py", title="Tailor Resume", url_path="tailor")

pg = st.navigation([home_page, tailor_page], position="hidden")

st.set_page_config(page_title="HIRE Ready AI", layout="wide")

with st.container():
    col1, col2, col3 = st.columns([1, 1, 4])
    
    with col1:
        st.page_link(home_page, label="Home", use_container_width=True)
    
    with col2:
        st.page_link(tailor_page, label="Tailor Resume", use_container_width=True)

    st.markdown("---")

pg.run()