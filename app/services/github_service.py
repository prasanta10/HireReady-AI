import os
import httpx
from fastapi import HTTPException
from dotenv import load_dotenv

load_dotenv()
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")

GITHUB_API_URL = "https://api.github.com"

def get_auth_header():
    if GITHUB_TOKEN:
        return {"Authorization": f"token {GITHUB_TOKEN}"}
    return {}

async def fetch_repo_readme(username: str, repo_name: str, client: httpx.AsyncClient):
    url = f"{GITHUB_API_URL}/repos/{username}/{repo_name}/readme"
    
    # Merge Auth token with the requirement to get raw text
    headers = get_auth_header()
    headers["Accept"] = "application/vnd.github.v3.raw"
    
    try:
        response = await client.get(url, headers=headers)
        if response.status_code == 200:
            return response.text
        else:
            return None 
    except Exception:
        return None

async def fetch_user_repos(username: str):
    headers = get_auth_header()
    
    async with httpx.AsyncClient() as client:
        print(f"DEBUG: Fetching repos for {username}...") # X-Ray Log
        
        response = await client.get(f"{GITHUB_API_URL}/users/{username}/repos", headers=headers)
        
        #Print the status code to the terminal
        print(f"DEBUG: GitHub Status Code: {response.status_code}")

        #Handle User Not Found
        if response.status_code == 404:
            raise HTTPException(status_code=404, detail="GitHub user not found")
            
        data = response.json()

        
        if isinstance(data, dict):
            # This catches "Rate Limit Exceeded" or "Bad Credentials"
            message = data.get("message", "Unknown Error")
            print(f"DEBUG: GitHub Error Message: {message}")
            raise HTTPException(status_code=response.status_code, detail=f"GitHub API Error: {message}")
            
        # Filter Forks
        original_repos = [repo for repo in data if repo.get("fork") is False]
        
        # Fetch Readmes for each repo
        results = []
        for repo in original_repos[:]:
            readme_content = await fetch_repo_readme(username, repo["name"], client)
            
            results.append({
                "name": repo["name"],
                "url": repo["html_url"],
                "language": repo["language"],
                "readme": readme_content
            })
            
        return results