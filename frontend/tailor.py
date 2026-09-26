import base64
import html

import requests
import streamlit as st

API_BASE = "http://127.0.0.1:8000"
API_URL = f"{API_BASE}/resume/tailor"
COVER_LETTER_URL = f"{API_BASE}/resume/cover-letter"

CHIP_CSS = """
<style>
.chip-row { display: flex; flex-wrap: wrap; gap: 6px; margin: 6px 0 14px 0; }
.chip {
    padding: 3px 10px; border-radius: 12px; font-size: 0.78rem;
    font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif; white-space: nowrap;
}
.chip-hit  { background: rgba(0, 200, 83, 0.15); color: #00C853; border: 1px solid #00C853; }
.chip-miss { background: rgba(255, 82, 82, 0.12); color: #FF7043; border: 1px solid #FF7043; }
</style>
"""


def display_pdf_from_bytes(file_bytes):
    """
    Helper to display a PDF from a raw byte string (for uploaded files).
    """
    base64_pdf = base64.b64encode(file_bytes).decode('utf-8')
    pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="600" type="application/pdf"></iframe>'
    st.markdown(pdf_display, unsafe_allow_html=True)


def display_pdf_from_base64(base64_string, height=800):
    """
    Helper to display a PDF directly from a Base64 string (for AI generated files).
    """
    pdf_display = f'<iframe src="data:application/pdf;base64,{base64_string}" width="100%" height="{height}" type="application/pdf"></iframe>'
    st.markdown(pdf_display, unsafe_allow_html=True)


def render_chips(terms, style="chip-hit", limit=30):
    if not terms:
        st.caption("None")
        return
    chips = "".join(
        f'<span class="chip {style}">{html.escape(str(term))}</span>' for term in terms[:limit]
    )
    extra = f'<span class="chip {style}">+{len(terms) - limit} more</span>' if len(terms) > limit else ""
    st.markdown(f'<div class="chip-row">{chips}{extra}</div>', unsafe_allow_html=True)


def render_ats_report(report, label):
    score = report.get("score", 0)
    st.markdown(f"#### {label}: {score}/100 &nbsp;·&nbsp; Grade {report.get('grade', '-')}")
    st.caption(report.get("verdict", ""))

    for item in report.get("breakdown", []):
        maximum = item.get("max", 1) or 1
        ratio = max(0.0, min(1.0, item.get("score", 0) / maximum))
        st.progress(ratio, text=f"{item['name']} — {item['score']:.1f}/{maximum} · {item['detail']}")

    st.markdown("**Keywords matched**")
    render_chips(report.get("matched_keywords", []), "chip-hit")

    st.markdown("**Keywords still missing**")
    render_chips(report.get("missing_keywords", []), "chip-miss")

    suggestions = report.get("suggestions", [])
    if suggestions:
        st.markdown("**How to score higher**")
        for tip in suggestions:
            st.markdown(f"- {tip}")


def render_ats_tab(result):
    before = result.get("ats_before") or {}
    after = result.get("ats_after") or {}
    delta = result.get("ats_delta") or {}

    if not after:
        st.info("No ATS report was returned by the backend.")
        return

    col1, col2, col3 = st.columns(3)
    col1.metric("Original Resume", f"{before.get('score', 0)}/100", help=before.get("verdict", ""))
    col2.metric(
        "Tailored Resume",
        f"{after.get('score', 0)}/100",
        delta=f"{delta.get('delta', 0):+.1f}",
        help=after.get("verdict", ""),
    )
    col3.metric("Grade", f"{before.get('grade', '-')} → {after.get('grade', '-')}")

    gained = delta.get("keywords_gained", [])
    if gained:
        st.markdown("**Keywords gained by tailoring**")
        render_chips(gained, "chip-hit")

    st.divider()
    render_ats_report(after, "Tailored resume score")

    with st.expander("See the original resume's score breakdown"):
        render_ats_report(before, "Original resume score")


def render_cover_letter_tab(result, job_description):
    st.caption("Generates a letter from the tailored resume, the job description and the matched projects.")

    with st.form("cover_letter_form"):
        col1, col2 = st.columns(2)
        with col1:
            company_name = st.text_input("Company Name", placeholder="e.g., Acme Corp")
            role_title = st.text_input("Role Title", placeholder="e.g., Backend Engineer")
        with col2:
            hiring_manager = st.text_input("Hiring Manager (optional)", placeholder="e.g., Alex Doe")
            tone = st.selectbox("Tone", ["professional", "enthusiastic", "concise", "formal"])
        generate = st.form_submit_button("Generate Cover Letter", type="primary")

    if generate:
        with st.spinner("Writing your cover letter..."):
            try:
                response = requests.post(
                    COVER_LETTER_URL,
                    json={
                        "resume_text": result.get("tailored_markdown", ""),
                        "job_description": job_description,
                        "company_name": company_name,
                        "role_title": role_title,
                        "hiring_manager": hiring_manager,
                        "tone": tone,
                        "projects": [
                            {
                                "name": match.get("name", ""),
                                "summary": match.get("summary", ""),
                                "url": match.get("url", ""),
                            }
                            for match in result.get("project_matches", [])
                        ],
                        "include_pdf": True,
                    },
                    timeout=300,
                )
                if response.status_code == 200:
                    st.session_state["cover_letter"] = response.json()
                else:
                    st.error(f"Cover letter failed: {response.text}")
            except Exception as exc:
                st.error(f"Connection Failed: {exc}")

    letter = st.session_state.get("cover_letter")
    if not letter:
        return

    markdown = letter.get("cover_letter_markdown", "")
    st.success(f"Cover letter ready ({letter.get('word_count', 0)} words).")

    col_dl1, col_dl2 = st.columns(2)
    with col_dl1:
        if letter.get("cover_letter_pdf_base64"):
            st.download_button(
                "📥 Download Cover Letter (PDF)",
                data=base64.b64decode(letter["cover_letter_pdf_base64"]),
                file_name="Cover_Letter.pdf",
                mime="application/pdf",
                type="primary",
                use_container_width=True,
                key="dl_cover_pdf",
            )
    with col_dl2:
        st.download_button(
            "📄 Download Markdown",
            data=markdown,
            file_name="Cover_Letter.md",
            mime="text/markdown",
            use_container_width=True,
            key="dl_cover_md",
        )

    preview_tab, source_tab = st.tabs(["Letter", "Markdown Source"])
    with preview_tab:
        st.markdown(markdown)
    with source_tab:
        st.code(markdown, language="markdown")


def render_results(result, job_description):
    st.divider()
    st.subheader("3. AI Optimization Results")

    tailored_markdown = result.get("tailored_markdown", "")
    pdf_base64 = result.get("final_pdf_base64", "")
    projects = result.get("selected_projects", [])
    after = result.get("ats_after") or {}

    tab1, tab2, tab3, tab4 = st.tabs(
        ["📄 PDF Preview", "📝 Markdown Source", "📊 ATS Score", "✉️ Cover Letter"]
    )

    with tab1:
        col_pdf_dl, col_pdf_view = st.columns([0.3, 0.7])

        with col_pdf_dl:
            st.markdown("### Download")

            if after:
                st.metric("ATS Score vs. JD", f"{after.get('score', 0)}/100", help=after.get("verdict", ""))
                if after.get("score", 0) < 70:
                    st.warning("Below 70 — check the ATS Score tab before sending this out.")

            st.download_button(
                label="📥 Download Tailored PDF",
                data=base64.b64decode(pdf_base64) if pdf_base64 else b"",
                file_name="Tailored_Resume.pdf",
                mime="application/pdf",
                type="primary",
                use_container_width=True,
                disabled=not pdf_base64,
                key="dl_resume_pdf",
            )
            st.success("✅ Ready for application")

            st.markdown("---")
            st.markdown("**Selected Projects:**")
            for match in result.get("project_matches") or [{"name": name} for name in projects]:
                score = match.get("match_score")
                suffix = f" · {score:.0f}% match" if score else ""
                st.markdown(f"- {match['name']}{suffix}")

            retrieval = result.get("retrieval") or {}
            if retrieval.get("backend"):
                st.caption(
                    f"Semantic index: {retrieval.get('backend')} · "
                    f"{retrieval.get('user_chunks', 0)} chunks · {retrieval.get('embedder', '')}"
                )

        with col_pdf_view:
            st.markdown("### Visual Preview")
            if pdf_base64:
                display_pdf_from_base64(pdf_base64)
            else:
                st.error("PDF generation failed.")

    with tab2:
        st.download_button(
            "📄 Download Markdown",
            data=tailored_markdown,
            file_name="Tailored_Resume.md",
            mime="text/markdown",
            key="dl_resume_md",
        )
        st.code(tailored_markdown, language="markdown")

    with tab3:
        render_ats_tab(result)

    with tab4:
        render_cover_letter_tab(result, job_description)


def show_tailor():
    st.markdown(CHIP_CSS, unsafe_allow_html=True)
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

    # --- RUN THE PIPELINE ---
    if submitted:
        if not username or not file or not job_description:
            st.warning("Action Required: Please complete all input fields.")
            return

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
                response = requests.post(API_URL, data=data, files=files, timeout=900)
                
                if response.status_code == 200:
                    # Cached in session state so the cover-letter form (which
                    # reruns the script) does not wipe the results.
                    st.session_state["tailor_result"] = response.json()
                    st.session_state["job_description"] = job_description
                    st.session_state.pop("cover_letter", None)
                    st.success("Optimization Complete!")
                else:
                    st.error(f"System Error: {response.text}")

            except Exception as e:
                st.error(f"Connection Failed: {e}")

    # --- RESULTS SECTION (Full Width, survives reruns) ---
    result = st.session_state.get("tailor_result")
    if result:
        render_results(result, st.session_state.get("job_description", ""))

# Run the page logic
show_tailor()
