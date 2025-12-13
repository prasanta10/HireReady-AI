import io
from fastapi import UploadFile
from pypdf import PdfReader

async def extract_text_from_pdf(file: UploadFile) -> str:
    content = await file.read()

    pdf_stream = io.BytesIO(content)
    reader = PdfReader(pdf_stream)
    text = ""
    
    for page in reader.pages:
        text += page.extract_text() or ""
        
    return text