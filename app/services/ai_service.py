from openai import OpenAI

client = OpenAI(
    base_url='http://localhost:11434/v1',
    api_key='ollama',
)

def summarize_project(readme_content: str, project_name: str) -> str:
    if not readme_content or not readme_content.strip():
        return f"Developed {project_name}; source code available on GitHub."
    
    if len(readme_content) < 50:
        return f"Created {project_name}, a Python-based utility."

    prompt = f"""
    You are an expert Technical Resume Writer.
    Rewrite the following GitHub README into a SINGLE, professional resume bullet point (max 30 words).
    Start with an action verb. Do not use "I". Output ONLY the bullet point.

    Project: {project_name}
    README:
    {readme_content[:2000]}
    """

    try:
        response = client.chat.completions.create(
            model="llama3.2", 
            messages=[
                {"role": "system", "content": "You are a helpful career assistant."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
        )
        
        return response.choices[0].message.content.strip()

    except Exception as e:
        print(f"Error calling Local LLM: {e}")
        return f"Developed {project_name} (Local AI Failed)"
    

def select_best_projects(projects: list, job_description: str) -> list:
    candidates_text = "\n".join(
        [f"- {p['name']}: {p['summary']}" for p in projects]
    )

    prompt = f"""
    You are a Technical Recruiter.
    
    JOB DESCRIPTION:
    {job_description[:1500]}
    
    CANDIDATE PROJECTS:
    {candidates_text}
    
    TASK:
    Identify the 3 projects from the list above that make the candidate look best for this specific Job Description.
    Return ONLY the names of the 3 projects, separated by commas.
    Example Output: project-alpha, data-viz-tool, api-gateway
    """

    try:
        response = client.chat.completions.create(
            model="llama3.2",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1, # Low temperature for precision    
        )
        selected_names = response.choices[0].message.content.strip()
        
        best_matches = [p for p in projects if p['name'] in selected_names]
        
        if not best_matches:
            return projects[:3]
            
        return best_matches[:3]

    except Exception as e:
        print(f"Selection Error: {e}")
        return projects[:3]

def generate_tailored_section(current_resume: str, best_projects: list, job_description: str) -> str:
    project_context = "\n".join(
        [f"- {p['name']}: {p['summary']}" for p in best_projects]
    )

    prompt = f"""
    You are an expert Resume Writer.
    
    JOB TARGET:
    {job_description[:1000]}
    
    CHOSEN PROJECTS:
    {project_context}
    
    TASK:
    Write a "Projects" section for a resume (in Markdown).
    - Use ONLY the Chosen Projects listed above.
    - Tailor the descriptions to use keywords from the Job Target.
    - Format as: "### Project Name \n - Bullet point..."
    - Do not include any intro or outro text.
    """
    
    try:
        response = client.chat.completions.create(
            model="llama3.2",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.5, # Medium temperature = Be creative but accurate
        )
        return response.choices[0].message.content.strip()
        
    except Exception:
        return "Error generating tailored section."