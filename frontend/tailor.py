import streamlit as st
import requests
import base64

API_URL = "http://127.0.0.1:8000/resume/tailor"

def display_pdf_from_bytes(file_bytes):
    """
    Helper to display a PDF from a raw byte string (for uploaded files).
    """
    base64_pdf = base64.b64encode(file_bytes).decode('utf-8')
    pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="600" type="application/pdf"></iframe>'
    st.markdown(pdf_display, unsafe_allow_html=True)

def display_pdf_from_base64(base64_string):
    """
    Helper to display a PDF directly from a Base64 string (for AI generated files).
    """
    pdf_display = f'<iframe src="data:application/pdf;base64,{base64_string}" width="100%" height="800" type="application/pdf"></iframe>'
    st.markdown(pdf_display, unsafe_allow_html=True)

def show_tailor():
    st.header("Resume Tailoring Interface")
    
    # --- LAYOUT: 2 Columns (Input vs Preview) ---
    col1, col2 = st.columns([0.4, 0.6], gap="medium")

    with col1:
        st.subheader("1. Configuration")
        st.caption("Provide your details to begin.")
        
        # File Uploader (Outside Form for instant preview)
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
        st.subheader("2. Original Document")
        if file is not None:
            # Show the uploaded file
            display_pdf_from_bytes(file.getvalue())
        else:
            st.info("Upload a PDF on the left to see the preview here.")

    # --- RESULTS SECTION (Full Width) ---
    if submitted:
        if not username or not file or not job_description:
            st.warning("Action Required: Please complete all input fields.")
            return

        st.divider()
        st.subheader("3. AI Optimization Results")
        
        with st.spinner("🤖 AI is rewriting your resume... (This takes about 15-30 seconds)"):
            try:
                # Prepare payload
                file.seek(0)
                files = {"file": (file.name, file, "application/pdf")}
                data = {
                    "username": username,
                    "job_description": job_description
                }
                
                # Call Backend
                response = requests.post(API_URL, data=data, files=files)
                
                if response.status_code == 200:
                    result = response.json()
                    st.success("Optimization Complete!")
                    
                    # 1. Extract Data
                    tailored_markdown = result.get("tailored_markdown", "")
                    pdf_base64 = result.get("final_pdf_base64", "")
                    projects = result.get("selected_projects", [])

                    # 2. Create Tabs for "Preview" vs "Raw Code"
                    tab1, tab2 = st.tabs(["📄 PDF Preview", "📝 Markdown Source"])
                    
                    with tab1:
                        col_pdf_dl, col_pdf_view = st.columns([0.3, 0.7])
                        
                        with col_pdf_dl:
                            st.markdown("### Download")
                            # Decode base64 to bytes for the download button
                            pdf_data = base64.b64decode(pdf_base64)
                            
                            st.download_button(
                                label="📥 Download Tailored PDF",
                                data=pdf_data,
                                file_name="Tailored_Resume.pdf",
                                mime="application/pdf",
                                type="primary",
                                use_container_width=True
                            )
                            st.success("✅ Ready for application")
                            
                            st.markdown("---")
                            st.markdown("**Selected Projects:**")
                            for p in projects:
                                st.markdown(f"- {p}")

                        with col_pdf_view:
                            st.markdown("### Visual Preview")
                            if pdf_base64:
                                display_pdf_from_base64(pdf_base64)
                            else:
                                st.error("PDF generation failed.")

                    with tab2:
                        st.code(tailored_markdown, language="markdown")
                        
                else:
                    st.error(f"System Error: {response.text}")
                    
            except Exception as e:
                st.error(f"Connection Failed: {e}")

# Run the page logic
show_tailor()