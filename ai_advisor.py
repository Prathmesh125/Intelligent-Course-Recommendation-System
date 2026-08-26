import requests

def generate_recommendation(query: str, courses: list, api_key: str) -> str:
    if not api_key:
        return "Please provide a valid Gemini API Key in the settings to use the AI Advisor."
    
    api_key = api_key.strip()
    if not api_key.startswith("AIzaSy"):
        return "⚠️ **Invalid API Key Format:** Gemini API keys generated from Google AI Studio must start with `AIzaSy`. It looks like you might be using an incorrect key or a Google Cloud Vertex AI token. Please generate a free key from [Google AI Studio](https://aistudio.google.com/app/apikey) and try again."
    
    # Format context
    context_str = ""
    if not courses:
        context_str = "No specific courses found in the database for this query."
    else:
        for i, c in enumerate(courses[:5]):
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

    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={api_key}"
    headers = {'Content-Type': 'application/json'}
    data = {
        "contents": [{"parts":[{"text": prompt}]}]
    }

    try:
        response = requests.post(url, headers=headers, json=data)
        if response.status_code == 404:
            return "⚠️ **API Access Denied (404 Not Found):** Google's servers rejected your API Key for this model. Ensure you generated your key from [Google AI Studio](https://aistudio.google.com/app/apikey) and that the Generative Language API is enabled on your account."
        
        response.raise_for_status()
        res_json = response.json()
        return res_json['candidates'][0]['content']['parts'][0]['text']
    except requests.exceptions.RequestException as e:
        return f"Error communicating with AI (Network): {str(e)}"
    except KeyError:
        return f"Error: Unexpected response format from AI. Please check your API key."
    except Exception as e:
        return f"Error communicating with AI: {str(e)}"
