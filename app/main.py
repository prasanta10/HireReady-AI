from fastapi import FastAPI

app = FastAPI(title="AI Resume Architect")

@app.get("/")
def read_root():
    return {"status": "active", "message": "The AI Resume Architect is online."}