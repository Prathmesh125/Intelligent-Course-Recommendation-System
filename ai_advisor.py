import google.generativeai as genai

def setup_gemini(api_key: str):
    genai.configure(api_key=api_key)

def generate_recommendation(query: str, courses: list, api_key: str) -> str:
    if not api_key:
        return "Please provide a valid Gemini API Key in the sidebar to use the AI Advisor."
    
    setup_gemini(api_key)
    
    # Format context
    context_str = ""
    if not courses:
        context_str = "No specific courses found in the database for this query."
    else:
        for i, c in enumerate(courses[:5]):
            # The courses come as dicts from pandas DataFrame rows usually
            title = c.get('title', 'Unknown Title')
            source = c.get('source', 'Unknown Platform')
            diff = c.get('difficulty', 'Unknown Difficulty')
            desc = str(c.get('description', 'No description'))[:200]
            url = c.get('url', '#')
            
            context_str += f"{i+1}. {title} ({source} | {diff})\n"
            context_str += f"Description: {desc}...\n"
            context_str += f"URL: {url}\n\n"
        
    prompt = f"""You are an expert AI Course Advisor for 'NLPRec', a learning recommendation platform. 
The user is asking for course advice.

User Query: "{query}"

Here are the top most relevant courses retrieved from our search engine:
{context_str}

Instructions:
1. Provide a friendly, conversational response.
2. Recommend the best course(s) from the list above and explain why they are a good fit for the user's specific query.
3. Use markdown (bolding, bullet points) to format your response nicely.
4. Do NOT say "based on the context provided". Act as if you natively know about these courses from our platform.
5. If the retrieved courses aren't a perfect match, politely explain what they are and why they might still be useful, or suggest what the user should search for instead.
"""

    try:
        model = genai.GenerativeModel('gemini-1.5-flash')
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Error communicating with AI: {str(e)}"
