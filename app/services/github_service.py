import httpx
from fastapi import HTTPException

GITHUB_API_URL = "https://api.github.com"

async def fetch_repo_readme(username: str, repo_name: str, client: httpx.AsyncClient):
    url = f"{GITHUB_API_URL}/repos/{username}/{repo_name}/readme"

    headers = {"Accept": "application/vnd.github.v3.raw"}
    
    try:
        response = await client.get(url, headers=headers)
        if response.status_code == 200:
            return response.text
        else:
            return None # No README found or other error
    except Exception:
        return None

async def fetch_user_repos(username: str):
    async with httpx.AsyncClient() as client:
    
        response = await client.get(f"{GITHUB_API_URL}/users/{username}/repos")
        
        if response.status_code == 404:
            raise HTTPException(status_code=404, detail="GitHub user not found")
            
        repos = response.json()
        
        #Filter out forks (Keep only original work)
        original_repos = [repo for repo in repos if repo.get("fork") is False]
        
        results = []
        for repo in original_repos:
            readme_content = await fetch_repo_readme(username, repo["name"], client)
            
            results.append({
                "name": repo["name"],
                "url": repo["html_url"],
                "description": repo["description"],
                "language": repo["language"],
                "readme": readme_content  # This is what the AI will read later!
            })
            
        return results