import httpx
from fastapi import HTTPException

GITHUB_API_URL = "https://api.github.com"

async def fetch_user_repos(username: str):
    async with httpx.AsyncClient() as client:

        response = await client.get(f"{GITHUB_API_URL}/users/{username}/repos")

        if response.status_code == 404:
            raise HTTPException(status_code=404, detail="GitHub user not found")
        
        if response.status_code != 200:
            raise HTTPException(status_code=response.status_code, detail="Error fetching data from GitHub")
            
        return response.json()