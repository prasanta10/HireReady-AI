"""Request/response models for the JSON endpoints."""
from typing import List, Optional

from pydantic import BaseModel, Field


class ProjectRef(BaseModel):
    name: str
    summary: str = ""
    url: str = ""


class ATSScoreRequest(BaseModel):
    resume_text: str = Field(..., description="Resume content (Markdown or plain text)")
    job_description: str
    is_markdown: bool = True


class CoverLetterRequest(BaseModel):
    resume_text: str = Field(..., description="Tailored resume Markdown, or the original resume text")
    job_description: str
    company_name: str = ""
    role_title: str = ""
    applicant_name: str = ""
    hiring_manager: str = ""
    tone: str = "professional"
    projects: List[ProjectRef] = []
    include_pdf: bool = True


class CoverLetterResponse(BaseModel):
    status: str
    cover_letter_markdown: str
    word_count: int
    cover_letter_pdf_base64: Optional[str] = None
