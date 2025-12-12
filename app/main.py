from fastapi import FastAPI
from app.services.github_service import fetch_user_repos

app = FastAPI(title="AI Resume Architect")

@app.get("/")
def read_root():
    return {"status": "active", "message": "The AI Resume Architect is online."}

@app.get("/github/{username}")
async def get_repos(username: str):
    repos = await fetch_user_repos(username)
    return {
        "username": username,
        "total_repos": len(repos),
        "repositories": [repo["name"] for repo in repos]
    }