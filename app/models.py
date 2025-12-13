from typing import Optional
from sqlmodel import SQLModel, Field
from datetime import datetime

class ProjectSummary(SQLModel, table=True):

    id: Optional[int] = Field(default=None, primary_key=True)

    user_id: str = Field(index=True)
    repo_name: str = Field(index=True)
    
    #ai-summary
    summary: str
    
    url: str
    language: Optional[str] = None
    
    updated_at: datetime = Field(default_factory=datetime.utcnow)