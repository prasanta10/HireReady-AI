import time
from fastapi import FastAPI, Depends
from contextlib import asynccontextmanager
from sqlalchemy.ext.asyncio import AsyncSession
from app.db import init_db, get_session
from app.services.github_service import fetch_user_repos
from app.services.ai_service import summarize_project
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

@app.get("/github/{username}")
async def get_repos(
    username: str, 
    db: AsyncSession = Depends(get_session) 
):
    repos = await fetch_user_repos(username)
    
    processed_repos = []
    for repo in repos[:5]:
        # Check if we have it in the Database
        existing_project = await crud.get_project(db, user_id=username, repo_name=repo["name"])
        
        if existing_project:
            print(f"DEBUG: Found '{repo['name']}' in DB! Skipping AI.")
            processed_repos.append(existing_project)
            
        else:
            print(f"DEBUG: '{repo['name']}' is new. Calling AI...")
            readme_text = repo.get("readme")
            summary = summarize_project(readme_text, repo["name"])
            
           
            new_project = ProjectSummary(
                user_id=username,
                repo_name=repo["name"],
                summary=summary,
                url=repo["url"],
                language=repo["language"]
            )
            await crud.create_project(db, new_project)
            
            processed_repos.append(new_project)
            
            time.sleep(1) 
            
    return {
        "username": username,
        "processed_count": len(processed_repos),
        "projects": processed_repos
    }