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

    # Raw README kept so the repo can be re-chunked/re-embedded without hitting
    # GitHub again; the hash tells us when the vector index is stale.
    readme: Optional[str] = None
    content_hash: Optional[str] = Field(default=None, index=True)
    indexed_at: Optional[datetime] = None

    updated_at: datetime = Field(default_factory=datetime.utcnow)
