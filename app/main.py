import time
from fastapi import FastAPI
from app.services.github_service import fetch_user_repos
from app.services.ai_service import summarize_project

app = FastAPI(title="AI Resume Architect")

@app.get("/")
def read_root():
    return {"status": "active", "message": "The AI Resume Architect is online."}

@app.get("/github/{username}")
async def get_repos(username: str):
    repos = await fetch_user_repos(username)
    
    processed_repos = []

    for repo in repos: 
        summary = summarize_project(repo["readme"], repo["name"])
        
        time.sleep(2)
        
        processed_repos.append({
            "name": repo["name"],
            "url": repo["url"],
            "language": repo["language"],
            "ai_summary": summary 
        })

    return {
        "username": username,
        "total_repos": len(repos),
        "repositories": processed_repos
    }