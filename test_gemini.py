import google.generativeai as genai

genai.configure(
    api_key="GELI_KEY_GAAGA_HALKAN"
)

model = genai.GenerativeModel("gemini-2.0-flash")

response = model.generate_content("Say hello")

print(response.text)
