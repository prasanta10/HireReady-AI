import ollama

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
        # FIXED: Use ollama.chat directly and move formatting options to 'options'
        response = ollama.chat(
            model="llama3.2", 
            messages=[
                {"role": "system", "content": "You are a helpful career assistant."},
                {"role": "user", "content": prompt}
            ],
            options={"temperature": 0.7}
        )
        
        # FIXED: Access response as a dictionary
        return response['message']['content'].strip()

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
        # FIXED: Use ollama.chat directly
        response = ollama.chat(
            model="llama3.2",
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0.1} # Low temp goes in options
        )
        # FIXED: Access response as dictionary
        selected_names = response['message']['content'].strip()
        
        best_matches = [p for p in projects if p['name'] in selected_names]
        
        if not best_matches:
            return projects[:3]
            
        return best_matches[:3]

    except Exception as e:
        print(f"Selection Error: {e}")
        return projects[:3]

def generate_tailored_resume(current_resume_text: str, best_projects: list, job_description: str) -> str:
    project_context = ""
    for p in best_projects:
        project_context += f"Project: {p['name']}\nDetails: {p['summary']}\nLink: {p['url']}\n\n"

    prompt = f"""
    You are an expert Resume Writer. 
    
    TASK: 
    Rewrite the candidate's resume to target a specific Job Description.
    1. Keep the Contact Info, Skills, and Experience sections mostly the same (just polish them).
    2. COMPLETELY REPLACE the "Projects" section with the 3 specific projects listed below.
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
    - Do NOT include any preamble like "Here is the resume". Just output the Markdown.
    """
    
    try:
        response = ollama.chat(model='llama3.2', messages=[{'role': 'user', 'content': prompt}])
        return response['message']['content'].strip()
    except Exception as e:
        return f"Error generating resume: {e}"