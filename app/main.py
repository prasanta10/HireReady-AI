import time
import io
import base64  # <--- Added this for encoding the PDF
from fastapi import FastAPI, Depends, UploadFile, File, Form, HTTPException
from contextlib import asynccontextmanager
from sqlalchemy.ext.asyncio import AsyncSession

# Import the new PDF generator function
from app.utils import extract_text_from_pdf, create_resume_pdf 
from app.db import init_db, get_session
from app.services.github_service import fetch_user_repos
# Import the "Full Rewrite" AI function
from app.services.ai_service import summarize_project, select_best_projects, generate_tailored_resume
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
    
    # 1. Extract Text from the uploaded PDF
    print("DEBUG: extracting PDF text...")
    resume_text = await extract_text_from_pdf(file)
   
    # 2. Fetch Repos from GitHub
    print(f"DEBUG: Fetching repos for {username}...")
    repos = await fetch_user_repos(username)
    
    project_summaries = []
    
    # SAFETY: Limit to first 10 repos to prevent freezing
    for repo in repos[:10]:
        existing = await crud.get_project(db, username, repo["name"])
        
        if existing:
            print(f"DEBUG: Found '{repo['name']}' in DB.")
            summary = existing.summary
            url = existing.url
        else:
            print(f"DEBUG: Summarizing '{repo['name']}' (New)...")
            summary = summarize_project(repo.get("readme"), repo["name"])
            url = repo.get("html_url", repo.get("url"))

            new_project = ProjectSummary(
                user_id=username,
                repo_name=repo["name"],
                summary=summary,
                url=url,
                # Use .get() to prevent crashes if language is missing
                language=repo.get("language") 
            )
            await crud.create_project(db, new_project)

            # Protect your local LLM from overheating/rate-limits
            time.sleep(1) 
            
        project_summaries.append({"name": repo["name"], "summary": summary, "url": url})
        
    # 3. Select Best Matches (The Brain 1)
    print("DEBUG: Selecting best projects...")
    best_projects = select_best_projects(project_summaries, job_description)
    
    # 4. Write Content (The Brain 2 - Full Resume Rewrite)
    print("DEBUG: Rewriting full resume...")
    tailored_markdown = generate_tailored_resume(resume_text, best_projects, job_description)
    
    # 5. Generate the New PDF
    print("DEBUG: Generating PDF file...")
    pdf_stream = create_resume_pdf(tailored_markdown)
    
    # 6. Encode PDF to Base64 (so we can send it in JSON)
    final_pdf_base64 = base64.b64encode(pdf_stream.getvalue()).decode('utf-8')

    return {
        "status": "success",
        "total_repos": len(repos),
        "selected_projects": [p['name'] for p in best_projects],
        "tailored_markdown": tailored_markdown,
        "final_pdf_base64": final_pdf_base64  # <--- The frontend will decode this to download
    }