
import os
import google.generativeai as genai

os.environ["GEMINI_API_KEY"] = "AIzaSyBbM8VtII2AsIlfAmbktRCleaBBB9RQTzo"  # Replace with your actual API key

genai.configure(api_key=os.environ["GEMINI_API_KEY"])

generation_config = {
    "temperature": 1,
    "top_p": 0.95,
    "top_k": 40,
    "max_output_tokens": 8192,
}

model = genai.GenerativeModel("gemini-2.0-flash-exp", generation_config=generation_config)
chat_session = model.start_chat(history=[])

def get_gemini_response(user_query):
    try:
        response = chat_session.send_message(user_query)
        if response.candidates and response.candidates[0].content.parts:
            text_output = ""
            for part in response.candidates[0].content.parts:
                if hasattr(part, "text"):
                    text_output += part.text
            # Remove Markdown list formatting (asterisks and leading/trailing spaces)
            cleaned_output = text_output.replace("* ", "").replace("*", "").strip()
            return cleaned_output
        else:
            return "No response content found."
    except Exception as e:
        return f"Error: {e}"

