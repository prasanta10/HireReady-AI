from typing import Dict, List, Optional, Sequence

import ollama

from app.config import FINAL_PROJECT_COUNT, LLM_MODEL


def _chat(prompt: str, system: Optional[str] = None, temperature: float = 0.7) -> str:
    """Single-turn call to the local Ollama model."""
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    response = ollama.chat(
        model=LLM_MODEL,
        messages=messages,
        options={"temperature": temperature},
    )
    return response["message"]["content"].strip()


def summarize_project(readme_content: str, project_name: str) -> str:
    if not readme_content or not readme_content.strip():
        return f"Developed {project_name}; source code available on GitHub."

    if len(readme_content) < 50:
        return f"Created {project_name}, a Python-based utility."

    prompt = f"""
    You are an expert Technical Resume Writer.
    Rewrite the following GitHub README into a THREE professional resume bullet point (max 30 words each).
    Use Markdown bullet point format.
    Start with an action verb. Do not use "I". Output ONLY the bullet point.

    Project: {project_name}
    README:
    {readme_content[:2000]}
    """

    try:
        return _chat(prompt, system="You are a helpful career assistant.", temperature=0.7)
    except Exception as e:
        print(f"Error calling Local LLM: {e}")
        return f"Developed {project_name} (Local AI Failed)"


def select_best_projects(
    projects: Sequence[Dict],
    job_description: str,
    count: int = FINAL_PROJECT_COUNT,
) -> List[Dict]:
    """Pick the final projects from a semantically retrieved shortlist.

    ``projects`` is expected to already be ordered by vector-search relevance
    (see ``retrieval_service.rank_repositories``); the LLM only re-ranks inside
    that shortlist, so a model failure degrades to the retrieval order rather
    than to an arbitrary slice.
    """
    projects = list(projects)
    if not projects:
        return []
    if len(projects) <= count:
        return projects

    candidates_text = "\n".join(
        f"- {p['name']} (semantic match {p.get('match_score', 0)}%): {p['summary']}"
        for p in projects
    )

    prompt = f"""
    You are a Technical Recruiter.

    JOB DESCRIPTION:
    {job_description[:1500]}

    CANDIDATE PROJECTS (pre-ranked by semantic similarity to the job description):
    {candidates_text}

    TASK:
    Identify the {count} projects from the list above that make the candidate look best for this
    specific Job Description. Prefer projects whose technology overlaps the job requirements.
    Return ONLY the names of the {count} projects, separated by commas.
    Example Output: project-alpha, data-viz-tool, api-gateway
    """

    try:
        selected_names = _chat(prompt, temperature=0.1)
        best_matches = [p for p in projects if p["name"] in selected_names]

        if not best_matches:
            return projects[:count]

        return best_matches[:count]

    except Exception as e:
        print(f"Selection Error: {e}")
        return projects[:count]


def generate_tailored_resume(
    current_resume_text: str,
    best_projects: Sequence[Dict],
    job_description: str,
) -> str:
    project_context = ""
    for p in best_projects:
        project_context += f"Project: {p['name']}\nDetails: {p['summary']}\nLink: {p['url']}\n"
        evidence = p.get("evidence")
        if evidence:
            project_context += f"Supporting detail from the repository:\n{evidence}\n"
        project_context += "\n"

    prompt = f"""
    You are an expert Resume Writer. 
    
    TASK: 
    Rewrite the candidate's resume to target a specific Job Description.
    1. Keep the Contact Info, Skills, and Experience sections mostly the same (just polish them).
    2. COMPLETELY REPLACE the "Projects" section with the {len(best_projects)} specific projects listed below.
    3. Output the result as clean, professional Markdown.
    
    INPUT CONTEXT:
    1. OLD RESUME TEXT:
    {current_resume_text}
    
    2. TARGET JOB DESCRIPTION:
    {job_description[:2000]}
    
    3. NEW PROJECTS TO INSERT (Use these exactly):
    {project_context}
    
    GUIDELINES:
    - Use H1 (#) for Name.
    - Use H2 (##) for Section Headers (Experience, Projects, Skills, Education).
    - Use bullet points (*) for details.
    - Ensure the "Projects" section uses STAR format and keywords from the JD.
    - Start every bullet with a strong action verb and quantify impact where the source allows.
    - Never invent employers, dates, degrees or metrics that are not in the source material.
    - Do NOT include any preamble like "Here is the resume". Just output the Markdown.
    """

    try:
        return _chat(prompt, temperature=0.7)
    except Exception as e:
        return f"Error generating resume: {e}"


def generate_cover_letter(
    resume_text: str,
    job_description: str,
    projects: Optional[Sequence[Dict]] = None,
    company_name: str = "",
    role_title: str = "",
    applicant_name: str = "",
    hiring_manager: str = "",
    tone: str = "professional",
) -> str:
    """Write a cover letter that matches the tailored resume and the target JD."""
    project_context = ""
    for p in projects or []:
        project_context += f"- {p['name']}: {p['summary']}\n"

    company = company_name.strip() or "the company named in the job description"
    role = role_title.strip() or "the role described in the job description"
    greeting = f"Dear {hiring_manager.strip()}," if hiring_manager.strip() else "Dear Hiring Manager,"
    signature = applicant_name.strip() or "the name found in the resume"

    prompt = f"""
    You are an expert career writer. Write a cover letter for this candidate.

    CANDIDATE RESUME:
    {resume_text[:4000]}

    TARGET JOB DESCRIPTION:
    {job_description[:2000]}

    HEADLINE PROJECTS TO REFERENCE:
    {project_context or "Use the strongest projects from the resume."}

    LETTER DETAILS:
    - Company: {company}
    - Role: {role}
    - Greeting to use: {greeting}
    - Sign off with: {signature}
    - Tone: {tone}

    RULES:
    - 4 paragraphs, 250-350 words total: hook, proof, fit, close.
    - Paragraph 2 must cite one or two concrete projects with their real results.
    - Mirror the vocabulary of the job description, but never copy sentences from it.
    - Never invent employers, degrees, dates or metrics that are absent from the resume.
    - Do not use placeholders such as [Your Name] or [Company]; use the details above.
    - Output clean Markdown only: no preamble, no explanation, no subject line.
    """

    try:
        return _chat(
            prompt,
            system="You are a concise, credible career writer who never fabricates experience.",
            temperature=0.6,
        )
    except Exception as e:
        return f"Error generating cover letter: {e}"
