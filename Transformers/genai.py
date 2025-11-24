from google import genai
model = genai.GenerativeModel("gemini-2.5-flash")
prompt = "Write a short summary of AI and its applications."
response = model.generate_content(prompt)
print(response.text)