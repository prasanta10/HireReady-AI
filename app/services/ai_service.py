from openai import OpenAI

client = OpenAI(
    base_url='http://localhost:11434/v1',
    api_key='ollama',
)

def summarize_project(readme_content: str, project_name: str) -> str:
    # 1. Validation (Keep our safety checks)
    if not readme_content or not readme_content.strip():
        return f"Developed {project_name}; source code available on GitHub."
    
    if len(readme_content) < 50:
        return f"Created {project_name}, a Python-based utility."

    #The Prompt
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