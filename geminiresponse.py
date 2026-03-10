# from google import genai

# # Configure client
# client = genai.Client(api_key="")


# def ask_gemini(prompt):
#     response = client.models.generate_content(
#         model="gemini-flash-latest",
#         contents=prompt
#     )
#     return response.text




# def chatbot():
#     print("Gemini Chatbot  (type 'exit' to quit)\n")

#     while True:
#         user_input = input("You: ")

#         if user_input.lower() == "exit":
#             break

#         gemini_reply = ask_gemini(user_input)

#         print("\n--- Gemini Response ---")
#         print(gemini_reply)
#         print("\n-----------------------\n")


# chatbot()


import os
from google import genai

# Read API key from environment variable
api_key = os.getenv("GEMINI_API_KEY")

# Configure client
client = genai.Client(api_key=api_key)


def ask_gemini(prompt):
    response = client.models.generate_content(
        model="gemini-flash-latest",
        contents=prompt
    )
    return response.text


def chatbot():
    print("Gemini Chatbot (type 'exit' to quit)\n")

    while True:
        user_input = input("You: ")

        if user_input.lower() == "exit":
            break

        gemini_reply = ask_gemini(user_input)

        print("\n--- Gemini Response ---")
        print(gemini_reply)
        print("\n-----------------------\n")


chatbot()
