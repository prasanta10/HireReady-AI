import time
from fastapi import FastAPI, Depends, UploadFile, File, Form, HTTPException
from contextlib import asynccontextmanager
from sqlalchemy.ext.asyncio import AsyncSession

from app.utils import extract_text_from_pdf
from app.db import init_db, get_session
from app.services.github_service import fetch_user_repos
from app.services.ai_service import summarize_project, select_best_projects, generate_tailored_section
from app import crud
from app.models import ProjectSummary

@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_db()
    print("DEBUG: Database tables created.")
    yield

app = FastAPI(title="AI Resume Architect", lifespan=lifespan)

@app.get("/")
def read_root():
    return {"status": "active", "message": "The AI Resume Architect is online."}

@app.post("/resume/tailor")
async def tailor_resume(
    file: UploadFile = File(...),         # The PDF
    username: str = Form(...),            # The GitHub Username
    job_description: str = Form(...),     # The Target JD
    db: AsyncSession = Depends(get_session)
):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only .pdf files allowed.")
    
    print("DEBUG: extracting PDF text...")
    resume_text = await extract_text_from_pdf(file)
   
    print(f"DEBUG: Fetching repos for {username}...")
    repos = await fetch_user_repos(username)
    
    project_summaries = []
    
    for repo in repos:
        existing = await crud.get_project(db, username, repo["name"])
        
        if existing:
            summary = existing.summary
            url = existing.url
        else:
            summary = summarize_project(repo.get("readme"), repo["name"])
            url = repo.get("html_url", repo.get("url"))

            new_project = ProjectSummary(
                user_id=username,
                repo_name=repo["name"],
                summary=summary,
                url=url,
                language=repo["language"]
            )
            await crud.create_project(db, new_project)

            time.sleep(1) 
            
        project_summaries.append({"name": repo["name"], "summary": summary, "url": url})
        
    # Select Best Matches (The Brain 1)
    print("DEBUG: Selecting best projects...")
    best_projects = select_best_projects(project_summaries, job_description)
    
    # Write Content (The Brain 2)
    print("DEBUG: Writing tailored content...")
    tailored_content = generate_tailored_section(resume_text, best_projects, job_description)
    
    return {
        "status": "success",
        "total_repos": len(repos),
        "selected_projects": [p['name'] for p in best_projects],
        "tailored_markdown": tailored_content
    }