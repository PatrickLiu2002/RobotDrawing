import os
from dotenv import load_dotenv
from google import genai
from google.genai import types

# Load variables from .env into the system environment
load_dotenv()

# Initialize the client. 
# It automatically detects GOOGLE_API_KEY from the environment.
client = genai.Client(
    http_options=types.HttpOptions(api_version="v1alpha")
)

def generate_cat_svg():
    # Use the 2026 preview model ID
    model_id = "gemini-3-flash-preview" 
    
    prompt = "Create a simple, minimalistic SVG outline of a house. Black stroke, no fill."

    config = types.GenerateContentConfig(
        system_instruction="Output ONLY raw SVG code. No markdown, no text.",
        temperature=0.2,
    )

    try:
        response = client.models.generate_content(
            model=model_id,
            contents=prompt,
            config=config
        )

        svg_code = response.text.strip()

        with open("house_from_env.svg", "w") as f:
            f.write(svg_code)

        print("Successfully generated house_from_env.svg using the .env key!")

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    generate_cat_svg()