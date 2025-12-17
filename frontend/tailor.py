import streamlit as st
import requests
import base64

API_URL = "http://127.0.0.1:8000/resume/tailor"

def display_pdf(uploaded_file):
    bytes_data = uploaded_file.getvalue()
    base64_pdf = base64.b64encode(bytes_data).decode('utf-8')
    pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="600" type="application/pdf"></iframe>'
    st.markdown(pdf_display, unsafe_allow_html=True)

def show_tailor():
    st.header("Resume Tailoring Interface")
    
    col1, col2 = st.columns([0.4, 0.6], gap="medium")

    with col1:
        st.subheader("1. Configuration")
        st.caption("Provide your details to begin.")
        
        file = st.file_uploader("Upload Resume (PDF)", type=["pdf"])

        with st.form("tailor_form"):
            username = st.text_input("GitHub Username", placeholder="e.g., octocat")
            
            job_description = st.text_area(
                "Job Description", 
                height=300, 
                placeholder="Paste the full job description here..."
            )
            
            submitted = st.form_submit_button("Analyze and Tailor", type="primary")

    with col2:
        st.subheader("2. Document Preview")
        if file is not None:
            display_pdf(file)
        else:
            st.info("Upload a PDF on the left to see the preview here.")
    if submitted:
        if not username or not file or not job_description:
            st.warning("Action Required: Please complete all input fields.")
            return

        with st.spinner("Processing... Extracting data and matching entities..."):
            try:
                file.seek(0)
                
                files = {"file": (file.name, file, "application/pdf")}
                data = {
                    "username": username,
                    "job_description": job_description
                }
                
                response = requests.post(API_URL, data=data, files=files)
                
                if response.status_code == 200:
                    result = response.json()
                    st.success("Optimization Complete!")
                    
                    st.divider()
                    
                    res_col1, res_col2 = st.columns(2)
                    
                    with res_col1:
                        st.subheader("Recommended Projects")
                        projects = result.get("selected_projects", [])
                        for p in projects:
                            st.markdown(f"**{p}**")
                    
                    with res_col2:
                        st.subheader("Tailored Content")
                        st.code(result.get("tailored_markdown", ""), language="markdown")
                else:
                    st.error(f"System Error: {response.text}")
                    
            except Exception as e:
                st.error(f"Connection Failed: {e}")
show_tailor()