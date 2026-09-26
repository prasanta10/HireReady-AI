import io
import re
from typing import Optional

from fastapi import UploadFile
from pypdf import PdfReader
from markdown_pdf import MarkdownPdf, Section

async def extract_text_from_pdf(file: UploadFile) -> str:
    content = await file.read()

    pdf_stream = io.BytesIO(content)
    reader = PdfReader(pdf_stream)
    text = ""

    for page in reader.pages:
        text += page.extract_text() or ""
        
    return text

def create_pdf_from_markdown(markdown_content: str, title: Optional[str] = None) -> io.BytesIO:
    """Render Markdown to a PDF byte stream."""
    pdf = MarkdownPdf(toc_level=0)
    meta = getattr(pdf, "meta", None)
    if title and isinstance(meta, dict):
        meta["title"] = title
    pdf.add_section(Section(markdown_content))
    pdf_output = io.BytesIO()
    pdf.save(pdf_output)
    pdf_output.seek(0)

    return pdf_output


def create_resume_pdf(markdown_content: str) -> io.BytesIO:
    return create_pdf_from_markdown(markdown_content, title="Tailored Resume")


def create_cover_letter_pdf(markdown_content: str) -> io.BytesIO:
    return create_pdf_from_markdown(markdown_content, title="Cover Letter")


def guess_applicant_name(resume_text: str) -> str:
    """Best-effort name pull from the top of a resume (Markdown H1 or first line)."""
    if not resume_text:
        return ""
    for line in resume_text.splitlines():
        candidate = line.strip().lstrip("#").strip(" *_")
        if not candidate:
            continue
        if "@" in candidate or re.search(r"\d{3}", candidate):
            continue
        words = candidate.split()
        if 1 < len(words) <= 5 and all(word[0].isalpha() for word in words if word):
            return candidate
        break
    return ""
