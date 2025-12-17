import io
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

def create_resume_pdf(markdown_content: str) -> io.BytesIO:
    pdf = MarkdownPdf(toc_level=0)
    pdf.add_section(Section(markdown_content))
    pdf_output = io.BytesIO()
    pdf.save(pdf_output)
    pdf_output.seek(0)
    
    return pdf_output