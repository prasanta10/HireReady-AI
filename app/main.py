import asyncio
import base64
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile
from sqlalchemy.ext.asyncio import AsyncSession

from app import crud
from app.config import FINAL_PROJECT_COUNT, LLM_COOLDOWN_SECONDS, MAX_REPOS, SHORTLIST_SIZE
from app.db import get_session, init_db
from app.schemas import ATSScoreRequest, CoverLetterRequest, CoverLetterResponse
from app.services import retrieval_service
from app.services.ai_service import (
    generate_cover_letter,
    generate_tailored_resume,
    select_best_projects,
    summarize_project,
)
from app.services.ats_service import compare_scores, score_resume
from app.services.github_service import fetch_user_repos
from app.utils import (
    create_cover_letter_pdf,
    create_resume_pdf,
    extract_text_from_pdf,
    guess_applicant_name,
)
from app.services.text_utils import word_count


@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_db()
    print("DEBUG: Database tables created.")
    # Warm the vector backend so the first request does not pay for it.
    await asyncio.to_thread(retrieval_service.index_stats)
    yield

app = FastAPI(title="AI Resume Architect", lifespan=lifespan)

@app.get("/")
def read_root():
    return {"status": "active", "message": "The AI Resume Architect is online."}


@app.get("/index/stats")
async def vector_index_stats(username: str | None = None):
    """Diagnostics for the semantic index (backend, embedder, chunk counts)."""
    return await asyncio.to_thread(retrieval_service.index_stats, username)

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
    
    # Make sure the vector index matches the embedding model currently in use.
    await asyncio.to_thread(retrieval_service.ensure_embedding_compatibility)

    project_summaries = []
    
    # SAFETY: Limit repos per request to prevent freezing
    for repo in repos[:MAX_REPOS]:
        existing = await crud.get_project(db, username, repo["name"])
        readme = repo.get("readme")

        if existing:
            print(f"DEBUG: Found '{repo['name']}' in DB.")
            summary = existing.summary
            url = existing.url
            language = existing.language
            readme = readme if readme is not None else existing.readme
        else:
            print(f"DEBUG: Summarizing '{repo['name']}' (New)...")
            # Ollama is synchronous and CPU-bound; keep the event loop free.
            summary = await asyncio.to_thread(summarize_project, readme, repo["name"])
            url = repo.get("html_url", repo.get("url"))
            language = repo.get("language")

            existing = await crud.upsert_project(
                db,
                user_id=username,
                repo_name=repo["name"],
                summary=summary,
                url=url,
                language=language,
                readme=readme,
            )

            # Protect your local LLM from overheating/rate-limits
            await asyncio.sleep(LLM_COOLDOWN_SECONDS)

        # 2b. Keep the vector index in sync with the repo's current content.
        fingerprint = retrieval_service.content_hash(summary, readme)
        if existing.content_hash != fingerprint or not retrieval_service.is_indexed(
            username, repo["name"], fingerprint
        ):
            print(f"DEBUG: Embedding '{repo['name']}' into the vector index...")
            try:
                await asyncio.to_thread(
                    retrieval_service.index_repository,
                    username,
                    repo["name"],
                    url,
                    language,
                    summary,
                    readme,
                )
                await crud.mark_indexed(db, existing, fingerprint)
            except Exception as exc:  # noqa: BLE001 - indexing must never break tailoring
                print(f"WARN: Failed to index '{repo['name']}': {exc}")

        project_summaries.append({"name": repo["name"], "summary": summary, "url": url})
        
    # 3. Semantic retrieval: shortlist repos whose content matches the JD
    print("DEBUG: Running semantic search over indexed repositories...")
    shortlist = await asyncio.to_thread(
        retrieval_service.rank_repositories,
        username,
        job_description,
        project_summaries,
        SHORTLIST_SIZE,
    )

    # 4. Select Best Matches (LLM re-rank inside the retrieved shortlist)
    print("DEBUG: Selecting best projects...")
    best_projects = await asyncio.to_thread(
        select_best_projects, shortlist, job_description, FINAL_PROJECT_COUNT
    )
    
    # 5. Write Content (Full Resume Rewrite)
    print("DEBUG: Rewriting full resume...")
    tailored_markdown = await asyncio.to_thread(
        generate_tailored_resume, resume_text, best_projects, job_description
    )
    
    # 6. Grade both versions against the JD before the user downloads anything
    print("DEBUG: Scoring resume against the job description...")
    ats_before = score_resume(resume_text, job_description, is_markdown=False)
    ats_after = score_resume(tailored_markdown, job_description, is_markdown=True)

    # 7. Generate the New PDF
    print("DEBUG: Generating PDF file...")
    pdf_stream = await asyncio.to_thread(create_resume_pdf, tailored_markdown)
    
    # 8. Encode PDF to Base64 (so we can send it in JSON)
    final_pdf_base64 = base64.b64encode(pdf_stream.getvalue()).decode('utf-8')

    return {
        "status": "success",
        "total_repos": len(repos),
        "selected_projects": [p['name'] for p in best_projects],
        "project_matches": [
            {
                "name": p["name"],
                "url": p.get("url", ""),
                "summary": p.get("summary", ""),
                "match_score": p.get("match_score", 0.0),
            }
            for p in best_projects
        ],
        "shortlist": [
            {"name": p["name"], "match_score": p.get("match_score", 0.0)} for p in shortlist
        ],
        "retrieval": retrieval_service.index_stats(username),
        "tailored_markdown": tailored_markdown,
        "ats_before": ats_before,
        "ats_after": ats_after,
        "ats_delta": compare_scores(ats_before, ats_after),
        "final_pdf_base64": final_pdf_base64  # <--- The frontend will decode this to download
    }


@app.post("/resume/ats-score")
async def ats_score(payload: ATSScoreRequest):
    """Grade any resume text against a job description (no LLM involved)."""
    if not payload.resume_text.strip():
        raise HTTPException(status_code=400, detail="resume_text is empty.")
    report = score_resume(
        payload.resume_text, payload.job_description, is_markdown=payload.is_markdown
    )
    return {"status": "success", "report": report}


@app.post("/resume/cover-letter", response_model=CoverLetterResponse)
async def cover_letter(payload: CoverLetterRequest):
    """Generate a cover letter that matches an already tailored resume."""
    if not payload.resume_text.strip():
        raise HTTPException(status_code=400, detail="resume_text is empty.")
    if not payload.job_description.strip():
        raise HTTPException(status_code=400, detail="job_description is empty.")

    applicant_name = payload.applicant_name or guess_applicant_name(payload.resume_text)

    markdown = await asyncio.to_thread(
        generate_cover_letter,
        payload.resume_text,
        payload.job_description,
        [project.model_dump() for project in payload.projects],
        payload.company_name,
        payload.role_title,
        applicant_name,
        payload.hiring_manager,
        payload.tone,
    )

    if markdown.startswith("Error generating cover letter"):
        raise HTTPException(status_code=502, detail=markdown)

    pdf_base64 = None
    if payload.include_pdf:
        pdf_stream = await asyncio.to_thread(create_cover_letter_pdf, markdown)
        pdf_base64 = base64.b64encode(pdf_stream.getvalue()).decode("utf-8")

    return CoverLetterResponse(
        status="success",
        cover_letter_markdown=markdown,
        word_count=word_count(markdown),
        cover_letter_pdf_base64=pdf_base64,
    )
